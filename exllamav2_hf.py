
from modules import shared
import random

from modules.text_generation import get_encoded_length, get_max_prompt_length

from modules.exllamav2_hf import Exllamav2HF

from exllamav2.generator import ExLlamaV2Sampler, ExLlamaV2StreamingGenerator
from exllamav2.cache import ExLlamaV2CacheBase
from exllamav2.model import _torch_device, ExLlamaV2
from exllamav2.compat import safe_move_tensor
from exllamav2 import (
    ExLlamaV2Cache,
    ExLlamaV2Cache_8bit,
    ExLlamaV2Lora,
    ExLlamaV2Config,
    ExLlamaV2Tokenizer
)
from exllamav2.attn import ExLlamaV2Attention

from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from modules.logging_colors import logger

import transformers
from modules.extensions import apply_extensions
from modules.text_generation import get_reply_from_output_ids, clear_torch_cache
from modules.grammar.grammar_utils import initialize_grammar
from modules.grammar.logits_process import GrammarConstrainedLogitsProcessor
from modules.callbacks import (
    Iteratorize,
    Stream,
    _StopEverythingStoppingCriteria
)
from transformers import LogitsProcessorList, is_torch_xpu_available
import time
import traceback
import pprint
import numpy as np

from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from exllamav2 import ext
from exllamav2.ext import exllamav2_ext as ext_c
import math
from torch import nn
from .util.general_stuff import *
# Detect flash-attn

has_flash_attn = False
try:
    import flash_attn
    flash_attn_ver = [int(t) for t in flash_attn.__version__.split(".") if t.isdigit()]
    is_ampere_or_newer_gpu = any(torch.cuda.get_device_properties(i).major >= 8 for i in range(torch.cuda.device_count()))
    
    if flash_attn_ver >= [2, 2, 1] and is_ampere_or_newer_gpu:
        from flash_attn import flash_attn_func
        has_flash_attn = True
except ModuleNotFoundError:
    pass

def get_model_info():
    info = {}
    
    info['last_kv_layer'] = shared.model.ex_model.last_kv_layer_idx
    info['head_layer'] = shared.model.ex_model.head_layer_idx
    
    info['attn_layers'] = []
    
    block_ct = 0
    for idx, module in enumerate(shared.model.ex_model.modules):
        module.block_idx = block_ct
        if isinstance(module, ExLlamaV2Attention):
            info['attn_layers'].append(idx)
            # it's not strictly true that every transformer block has 1 attention layer
            # so we can't actually reliably use this as a proxy for block index
            # however -- and hear me out on this one -- let's do it anyway.
            block_ct += 1 
    info['block_ct'] = block_ct
    info['layers_count'] = info['head_layer'] + 1
    
    return info

def remove_chip():
    if hasattr(shared.model, 'hackingchip'):
        delattr(shared.model, 'hackingchip')
    if hasattr(shared.model.ex_model, 'hackingchip'):
        delattr(shared.model.ex_model, 'hackingchip')

def hijack_loader(hackingchip):
    shared.model.hackingchip = hackingchip # Putting it here too, this might be the main place eventually
    shared.model.ex_model.hackingchip = hackingchip # hackingchip installed
    shared.model.model_info = get_model_info()
    
    if hackingchip.prompts.batch_size != shared.model.ex_cache.batch_size: # the hackingchip tends to have extra batches, so it's time to prepare for that
        # I'm not correctly deleting the existing cache, but it gets removed from VRAM somehow anyway
        if shared.args.cache_8bit:
            shared.model.ex_cache = ExLlamaV2Cache_8bit(shared.model.ex_model, hackingchip.prompts.batch_size)
        else:
            shared.model.ex_cache = ExLlamaV2Cache(shared.model.ex_model, hackingchip.prompts.batch_size)
        
    # Hijack functions
    shared.model.ex_model._forward = hijack_model_forward.__get__(shared.model.ex_model, ExLlamaV2)
    
    # May need to do shared.model.sample (sample from GenerationMixIn)
    # And it would only be to copy over the index 0 tokens over the rest lol, but it's necessary
    shared.model.sample = hijack_sample.__get__(shared.model, Exllamav2HF)

    # Call has to be hijacked on the class level
    Exllamav2HF.__call__ = hijack_call.__get__(shared.model, Exllamav2HF)
    
    # The only way I could see to get this to work also requires hooking the tokenizer
    shared.tokenizer._encode_plus = hijack_encode_plus.__get__(shared.tokenizer, PreTrainedTokenizerFast)
    
    # You'd think using ooba's hooks would make life easier, but they actually make life harder lol
    hackingchip.custom_generate_reply = hijack_generate_reply_HF
        
    # Hooking attention
    for idx, module in enumerate(shared.model.ex_model.modules):
        if isinstance(module, ExLlamaV2Attention):
            module.forward = hijack_attn_forward.__get__(module, ExLlamaV2Attention)
            
# The below functions come from ooba and exllamav2, my code is just inserted into them (anything dealing with hackingchip)

from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.tokenization_utils_base import (
    INIT_TOKENIZER_DOCSTRING,
    AddedToken,
    BatchEncoding,
    PreTokenizedInput,
    PreTokenizedInputPair,
    PreTrainedTokenizerBase,
    SpecialTokensMixin,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)
from transformers.utils import PaddingStrategy, add_end_docstrings, logging

# Ok, after getting the tokenizing figured out, this is the main thing to change to implement BHC in HF
def hijack_call(self, *args, **kwargs):
    use_cache = kwargs.get('use_cache', True)
    labels = kwargs.get('labels', None)
    past_key_values = kwargs.get('past_key_values', None)
    
    hackingchip = shared.model.hackingchip if hasattr(shared.model, 'hackingchip') and shared.model.hackingchip.ui_settings['on'] else None

    # I'm just assuming no one is using the default negative cfg with this, I'm pretty sure everything will break if you try

    if len(args) > 0: # shouldn't be doing this anyway
        if not shared.args.cfg_cache:
            logger.error("Please enable the cfg-cache option to use CFG with ExLlamav2_HF.")
            return

        input_ids = args[0]
        is_negative = True
        past_seq = self.past_seq_negative
        ex_cache = self.ex_cache_negative
    else:
        input_ids = kwargs['input_ids']
        is_negative = False
        past_seq = self.past_seq
        ex_cache = self.ex_cache

    seq = input_ids.tolist()
    if is_negative and past_key_values is not None:
        seq = past_key_values + seq

    seq_tensor = torch.tensor(seq)
    reset = True
    
    # Make the forward call
    if labels is None:
        if past_seq is not None:
            min_length = min(past_seq.shape[1], seq_tensor.shape[1])
            indices = torch.nonzero(~torch.eq(past_seq[:, :min_length], seq_tensor[:, :min_length]))
            if indices.shape[0] > 0:
                longest_prefix = torch.min(indices[0], dim=0).item() # this probably needs to be min, right?
            else:
                longest_prefix = min_length

            if longest_prefix > 0:
                reset = False
                ex_cache.current_seq_len = longest_prefix
                if seq_tensor.shape[1] - longest_prefix > 1:
                    self.ex_model.forward(seq_tensor[:, longest_prefix:-1], ex_cache, preprocess_only=True, loras=self.loras)
                elif seq_tensor.shape[1] == longest_prefix:
                    # Very tricky: if the prefix we are reusing *is* the input_ids, then we have to back up the cache pointer by one,
                    # because we feed input_ids[-1] to forward() below, but that last token is already in the cache!
                    ex_cache.current_seq_len -= 1

        if reset:
            ex_cache.current_seq_len = 0
            if seq_tensor.shape[1] > 1:
                self.ex_model.forward(seq_tensor[:, :-1], ex_cache, preprocess_only=True, loras=self.loras)

        logits = self.ex_model.forward(seq_tensor[:, -1:], ex_cache, loras=self.loras).to(input_ids.device).float()
    else:
        ex_cache.current_seq_len = 0
        logits = self.ex_model.forward(seq_tensor, ex_cache, last_id_only=False, loras=self.loras).float()

    if is_negative:
        self.past_seq_negative = seq_tensor
    else:
        self.past_seq = seq_tensor

    # I'm not really sure here, I'm going to hope the below code will just work with multiple batches

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, logits.shape[-1])
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        
    if hackingchip:
        for chip_settings in hackingchip.settings:
            if chip_settings.logits_settings:
                if chip_settings.logits_settings.cfg_func:
                    logits = chip_settings.logits_settings.cfg_func(logits, chip_settings.logits_settings, hackingchip)
                else:
                    print("cfg_func required")
                    
    return CausalLMOutputWithPast(logits=logits, past_key_values=seq if use_cache else None, loss=loss)

# Unfortunately, I see no other way than hooking part of transformers too, this is hopefully the only transformers func needed
def hijack_encode_plus(
    self,
    text: Union[TextInput, PreTokenizedInput],
    text_pair: Optional[Union[TextInput, PreTokenizedInput]] = None,
    add_special_tokens: bool = True,
    padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
    truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
    max_length: Optional[int] = None,
    stride: int = 0,
    is_split_into_words: bool = False,
    pad_to_multiple_of: Optional[int] = None,
    return_tensors: Optional[bool] = None,
    return_token_type_ids: Optional[bool] = None,
    return_attention_mask: Optional[bool] = None,
    return_overflowing_tokens: bool = False,
    return_special_tokens_mask: bool = False,
    return_offsets_mapping: bool = False,
    return_length: bool = False,
    verbose: bool = True,
    **kwargs,
) -> BatchEncoding:
    # The only reason I need to hook transformers is because of how batched_input can't handle a text list
    if isinstance(text, list):
        # Should I even do text_pair here? I would have to pair it with each entry of text in the list?
        batched_input = text
        
        # I'm going to just override this and force it to use the strategies I want here because passing it in is problematic
        padding_strategy = PaddingStrategy.LONGEST
        truncation_strategy = TruncationStrategy.LONGEST_FIRST
    else:
        batched_input = [(text, text_pair)] if text_pair else [text]
    batched_output = self._batch_encode_plus(
        batched_input,
        is_split_into_words=is_split_into_words,
        add_special_tokens=add_special_tokens,
        padding_strategy=padding_strategy,
        truncation_strategy=truncation_strategy,
        max_length=max_length,
        stride=stride,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors=return_tensors,
        return_token_type_ids=return_token_type_ids,
        return_attention_mask=return_attention_mask,
        return_overflowing_tokens=return_overflowing_tokens,
        return_special_tokens_mask=return_special_tokens_mask,
        return_offsets_mapping=return_offsets_mapping,
        return_length=return_length,
        verbose=verbose,
        **kwargs,
    )

    # Return tensor is None, then we can remove the leading batch axis
    # Overflowing tokens are returned as a batch of output so we keep them in this case
    if return_tensors is None and not return_overflowing_tokens:
        batched_output = BatchEncoding(
            {
                key: value[0] if len(value) > 0 and isinstance(value[0], list) else value
                for key, value in batched_output.items()
            },
            batched_output.encodings,
        )

    self._eventual_warn_about_too_long_sequence(batched_output["input_ids"], max_length, verbose)

    return batched_output

def hijack_generate_reply_HF(question, original_question, seed, state, stopping_strings=None, is_chat=False):
    generate_params = {}
    for k in ['max_new_tokens', 'temperature', 'temperature_last', 'dynamic_temperature', 'dynatemp_low', 'dynatemp_high', 'dynatemp_exponent', 'top_p', 'min_p', 'top_k', 'repetition_penalty', 'presence_penalty', 'frequency_penalty', 'repetition_penalty_range', 'typical_p', 'tfs', 'top_a', 'guidance_scale', 'penalty_alpha', 'mirostat_mode', 'mirostat_tau', 'mirostat_eta', 'do_sample', 'encoder_repetition_penalty', 'no_repeat_ngram_size', 'min_length', 'num_beams', 'length_penalty', 'early_stopping']:
        generate_params[k] = state[k]

    if state['negative_prompt'] != '':
        generate_params['negative_prompt_ids'] = encode(state['negative_prompt'])

    if state['prompt_lookup_num_tokens'] > 0:
        generate_params['prompt_lookup_num_tokens'] = state['prompt_lookup_num_tokens']

    for k in ['epsilon_cutoff', 'eta_cutoff']:
        if state[k] > 0:
            generate_params[k] = state[k] * 1e-4

    if state['ban_eos_token']:
        generate_params['suppress_tokens'] = [shared.tokenizer.eos_token_id]

    if state['custom_token_bans']:
        to_ban = [int(x) for x in state['custom_token_bans'].split(',')]
        if len(to_ban) > 0:
            if generate_params.get('suppress_tokens', None):
                generate_params['suppress_tokens'] += to_ban
            else:
                generate_params['suppress_tokens'] = to_ban

    generate_params.update({'use_cache': not shared.args.no_cache})
    if shared.args.deepspeed:
        generate_params.update({'synced_gpus': True})
        
    hackingchip = shared.model.hackingchip if hasattr(shared.model, 'hackingchip') and shared.model.hackingchip.ui_settings['on'] else None
    if hackingchip:
        if hasattr(hackingchip.prompts, 'batch_prompts'):
            question = hackingchip.prompts.batch_prompts
            
    # Encode the input
    input_ids = encode(question, add_bos_token=state['add_bos_token'], truncation_length=get_max_prompt_length(state))
    output = input_ids[0]
    cuda = not any((shared.args.cpu, shared.args.deepspeed))
    if state['auto_max_new_tokens']:
        generate_params['max_new_tokens'] = state['truncation_length'] - input_ids.shape[-1]

    # Add the encoded tokens to generate_params
    question, input_ids, inputs_embeds = apply_extensions('tokenizer', state, question, input_ids, None)
    original_input_ids = input_ids
    generate_params.update({'inputs': input_ids})
    if inputs_embeds is not None:
        generate_params.update({'inputs_embeds': inputs_embeds})

    # Stopping criteria / eos token
    eos_token_ids = [shared.tokenizer.eos_token_id] if shared.tokenizer.eos_token_id is not None else []
    generate_params['eos_token_id'] = eos_token_ids
    generate_params['stopping_criteria'] = transformers.StoppingCriteriaList()
    generate_params['stopping_criteria'].append(_StopEverythingStoppingCriteria())

    # Logits processor
    processor = state.get('logits_processor', LogitsProcessorList([]))
    if not isinstance(processor, LogitsProcessorList):
        processor = LogitsProcessorList([processor])

    # Grammar
    if state['grammar_string'].strip() != '':
        grammar = initialize_grammar(state['grammar_string'])
        grammar_processor = GrammarConstrainedLogitsProcessor(grammar)
        processor.append(grammar_processor)

    apply_extensions('logits_processor', processor, input_ids)
    generate_params['logits_processor'] = processor

    if shared.args.verbose:
        logger.info("GENERATE_PARAMS=")
        filtered_params = {key: value for key, value in generate_params.items() if not isinstance(value, torch.Tensor)}
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(filtered_params)
        print()

    t0 = time.time()
    try:
        if not is_chat and not shared.is_seq2seq:
            yield ''

        # Generate the entire reply at once.
        if not state['stream']:
            with torch.no_grad():
                output = shared.model.generate(**generate_params)[0]
                if cuda:
                    output = output.cuda()

            starting_from = 0 if shared.is_seq2seq else len(input_ids[0])
            yield get_reply_from_output_ids(output, state, starting_from=starting_from)

        # Stream the reply 1 token at a time.
        # This is based on the trick of using 'stopping_criteria' to create an iterator.
        else:

            def generate_with_callback(callback=None, *args, **kwargs):
                kwargs['stopping_criteria'].append(Stream(callback_func=callback))
                clear_torch_cache()
                with torch.no_grad():
                    shared.model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(generate_with_callback, [], kwargs, callback=None)

            with generate_with_streaming(**generate_params) as generator:
                cumulative_reply = ''
                starting_from = 0 if shared.is_seq2seq else len(input_ids[0])
                for output in generator:
                    if output[-1] in eos_token_ids:
                        break

                    new_content = get_reply_from_output_ids(output, state, starting_from=starting_from)
                    # check the partial unicode character
                    if chr(0xfffd) in new_content:
                        continue

                    cumulative_reply += new_content
                    starting_from = len(output)
                    yield cumulative_reply

    except Exception:
        traceback.print_exc()
    finally:
        t1 = time.time()
        original_tokens = len(original_input_ids[0])
        new_tokens = len(output) - (original_tokens if not shared.is_seq2seq else 0)
        print(f'Output generated in {(t1-t0):.2f} seconds ({new_tokens/(t1-t0):.2f} tokens/s, {new_tokens} tokens, context {original_tokens}, seed {seed})')
        return

def encode(prompt, add_special_tokens=True, add_bos_token=True, truncation_length=None):
    if shared.tokenizer is None:
        raise ValueError('No tokenizer is loaded')
    
    if isinstance(prompt, list):
        input = [str(element) for element in prompt]
    else:
        input = str(prompt)
    
    if shared.model.__class__.__name__ in ['LlamaCppModel', 'CtransformersModel', 'Exllamav2Model']:
        input_ids = shared.tokenizer.encode(input, max_length=truncation_length)
        if shared.model.__class__.__name__ not in ['Exllamav2Model']:
            input_ids = np.array(input_ids).reshape(1, len(input_ids))
    else:
        input_ids = shared.tokenizer.encode(input, return_tensors='pt', add_special_tokens=add_special_tokens, max_length=truncation_length)
        if not add_bos_token:
            while len(input_ids[0]) > 0 and input_ids[0][0] == shared.tokenizer.bos_token_id:
                input_ids = input_ids[:, 1:]

    # Handling truncation
    if truncation_length is not None:
        input_ids = input_ids[:, -truncation_length:]

    if shared.model.__class__.__name__ in ['LlamaCppModel', 'Exllamav2Model', 'CtransformersModel'] or shared.args.cpu:
        return input_ids
    elif shared.args.deepspeed:
        return input_ids.to(device=local_rank)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        return input_ids.to(device)
    elif is_torch_xpu_available():
        return input_ids.to("xpu:0")
    else:
        return input_ids.cuda()
    
# I have to copy this huge sample function just to change like one line!
    
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)

from transformers.generation import GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput
GenerateNonBeamOutput = Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput]

import warnings
import torch.distributed as dist

def hijack_sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    <Tip warning={true}>

    In most cases, you do not need to call [`~generation.GenerationMixin.sample`] directly. Use generate() instead.
    For an overview of generation strategies and code examples, check the [following
    guide](../generation_strategies).

    </Tip>

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        logits_warper (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
            to warp the prediction score distribution of the language modeling head applied before multinomial
            sampling at each generation step.
        max_length (`int`, *optional*, defaults to 20):
            **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
            tokens. The maximum length of the sequence to be generated.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`Union[int, List[int]]`, *optional*):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
        A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.

    Examples:

    ```python
    >>> from transformers import (
    ...     AutoTokenizer,
    ...     AutoModelForCausalLM,
    ...     LogitsProcessorList,
    ...     MinLengthLogitsProcessor,
    ...     TopKLogitsWarper,
    ...     TemperatureLogitsWarper,
    ...     StoppingCriteriaList,
    ...     MaxLengthCriteria,
    ... )
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

    >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
    >>> model.config.pad_token_id = model.config.eos_token_id
    >>> model.generation_config.pad_token_id = model.config.eos_token_id

    >>> input_prompt = "Today is a beautiful day, and"
    >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

    >>> # instantiate logits processors
    >>> logits_processor = LogitsProcessorList(
    ...     [
    ...         MinLengthLogitsProcessor(15, eos_token_id=model.generation_config.eos_token_id),
    ...     ]
    ... )
    >>> # instantiate logits processors
    >>> logits_warper = LogitsProcessorList(
    ...     [
    ...         TopKLogitsWarper(50),
    ...         TemperatureLogitsWarper(0.7),
    ...     ]
    ... )

    >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

    >>> torch.manual_seed(0)  # doctest: +IGNORE_RESULT
    >>> outputs = model.sample(
    ...     input_ids,
    ...     logits_processor=logits_processor,
    ...     logits_warper=logits_warper,
    ...     stopping_criteria=stopping_criteria,
    ... )

    >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ['Today is a beautiful day, and we must do everything possible to make it a day of celebration.']
    ```"""
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False  # used by synced_gpus only
    # auto-regressive generation
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # sample
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        
        # I copied this entire function just to put this one line into it
        # Not even checking if hackingchip is active, but do I need to?
        next_tokens[1:] = next_tokens[0]

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids

@torch.inference_mode()
def hijack_model_forward(self,
                input_ids,
                cache = None,
                input_mask = None,
                preprocess_only = False,
                last_id_only = False,
                loras = None,
                return_last_state = False,
                position_offsets = None):

    batch_size, seq_len = input_ids.shape
    past_len = 0
    if cache is not None:
        if isinstance(cache, ExLlamaV2CacheBase):
            past_len = cache.current_seq_len
        else:
            past_len = [c.current_seq_len for c in cache]

    # assert cache is None or isinstance(cache, list) or batch_size <= cache.batch_size

    x = input_ids
    attn_params = ExLlamaV2Attention.Params(batch_size, seq_len, past_len, input_mask, position_offsets)
    last_state = None
    
    hackingchip = self.hackingchip if hasattr(self, 'hackingchip') else None

    for idx, module in enumerate(self.modules):

        device = _torch_device(module.device_idx)

        # Onward

        if idx == self.head_layer_idx:
            if last_id_only and return_last_state:
                x = x.narrow(-2, -1, 1)
                last_state = x
            elif last_id_only:
                x = x.narrow(-2, -1, 1)
            elif return_last_state:
                last_state = x.narrow(-2, -1, 1)

        x = safe_move_tensor(x, device)
        x = module.forward(x, cache = cache, attn_params = attn_params, past_len = past_len, loras = loras)
                  
        # Even if the attention layers feature makes this mostly redundant, it is still useful in various ways
        if hackingchip:
            for chip_settings in hackingchip.settings:
                if chip_settings.layer_settings[idx] != None:
                    settings = chip_settings.layer_settings[idx]
                    
                    if settings.cfg_func:
                        x = settings.cfg_func(x, settings, hackingchip)
                        None
                    else:
                        print("cfg_func required")
                                                
        if preprocess_only and idx == self.last_kv_layer_idx:
            x = None
            break

    # Advance cache

    if cache is not None:
        if isinstance(cache, list):
            for c in cache: c.current_seq_len += seq_len
        else:
            cache.current_seq_len += seq_len

    # Set padding logits to -inf

    if x is not None:
        head_padding = self.modules[-1].padding
        if head_padding > 0:
            x[:, :, -head_padding:] = -65504.

    return x, last_state

hidden_state_diminfo = {'batch_size' : 0, 'seq_vec': 1, 'vec_component': 2}
unflash_att_diminfo = {'batch_size' : 0, 'head': 1, 'seq_vec': 2, 'vec_component': 3}
flash_att_diminfo = {'batch_size' : 0, 'seq_vec': 1, 'head': 2, 'vec_component': 3}

def hijack_attn_forward(self, hidden_states, cache = None, attn_params = None, past_len = None, intermediates = False, loras = None):
    global has_flash_attn

    def hack_states(states, states_settings, dim_info=None):
        if states_settings.cfg_func:
            states = states_settings.cfg_func(states, states_settings, hackingchip, 
                                                  layer_idx=self.layer_idx,
                                                  block_idx=self.block_idx,
                                                  dim_info=dim_info,
                                                  model = shared.model,
                                                  module = self,
                                                  attn_params = attn_params,
                                                  past_len = past_len, 
                                                  total_layers=hackingchip.attn_count, # storing attn_count on hackingchip
                                                  total_blocks=shared.model.model_info['block_ct'],
                                                  cache=cache)
        else:
            print("cfg_func required")
            # if hackingchip.prompts.numneg > 0 and states_settings.weight != 0.0:
            #     state_neg_steering = states[hackingchip.prompts.numpos:hackingchip.prompts.negend]
            #     state_neg_steering = torch.mean(state_neg_steering, dim=0, keepdim=False)
            #     state_neg_steering = states_settings.weight * (state_neg_steering - states[0])
                
            #     states -= state_neg_steering #I think the lines since my previous comment should be moved to a dedicated chip.
        return states
    
    #Hacking chip stuff
    hackingchip = shared.model.ex_model.hackingchip if hasattr(shared.model.ex_model, 'hackingchip') else None
    settings = [chip.attn_settings[self.layer_idx] for chip in hackingchip.settings if chip.attn_settings[self.layer_idx] is not None] if hackingchip else []
    
    for chip_settings in settings:
        if chip_settings.attn_mask: attn_mask = hack_states(hidden_states, chip_settings.attn_mask, None)
          
    #Hacking chip stuff
    for chip_settings in settings:
        if chip_settings.h: hidden_states = hack_states(hidden_states, chip_settings.h, dim_info=hidden_state_diminfo)
    
    if self.q_handle is None or intermediates:
        return self.forward_torch(hidden_states, cache, attn_params, past_len, intermediates, loras = loras)

    batch_size = hidden_states.shape[0]
    q_len = hidden_states.shape[1]

    direct = (batch_size == 1 and cache is not None and isinstance(cache, ExLlamaV2CacheBase))

    # past_len = 0
    # if cache is not None:
    #     if isinstance(cache, ExLlamaV2Cache):
    #         past_len = cache.current_seq_len
    #     if isinstance(cache, list):
    #         past_len = [c.current_seq_len for c in cache]

    num_attention_heads = self.model.config.num_attention_heads
    num_key_value_heads = self.model.config.num_key_value_heads
    num_key_value_groups = self.model.config.num_key_value_groups
    head_dim = self.model.config.head_dim
    hidden_size = self.model.config.hidden_size

    constants = self.model.get_device_tensors(self.device_idx)

    q_shape = hidden_states.shape[:-1] + (self.q_proj.out_features,)
    k_shape = hidden_states.shape[:-1] + (self.k_proj.out_features,)
    v_shape = hidden_states.shape[:-1] + (self.v_proj.out_features,)
    q_states = torch.empty(q_shape, device = hidden_states.device, dtype = torch.half)

    # If conditions are right we can write the K/V projections directly into the cache

    if direct:

        batch_keys, batch_values = cache.get_kv_state(self.layer_idx, batch_size, 0, past_len)
        k_states = batch_keys.narrow(0, 0, batch_size).narrow(1, past_len, q_len)
        v_states = batch_values.narrow(0, 0, batch_size).narrow(1, past_len, q_len)

    else:

        k_states = torch.empty(k_shape, device = hidden_states.device, dtype = torch.half)
        v_states = torch.empty(v_shape, device = hidden_states.device, dtype = torch.half)

    # RMS norm, Q/K/V projections, position embeddings

    if loras is None or self.temp_lora_size == 0:
        pass_loras = []
        pass_lora_temp = ext.none_tensor
    else:
        pass_loras = [id(x) for x in loras]
        pass_lora_temp = torch.empty((self.temp_lora_size,), dtype = torch.half, device = hidden_states.device)

    if attn_params.multi_cache:
        pass_past_len_1 = -1
        pass_past_len_2 = attn_params.get_past_lens(hidden_states.device)
    elif attn_params.position_offsets is not None:
        pass_past_len_1 = past_len
        pass_past_len_2 = attn_params.get_position_offsets(hidden_states.device)
    else:
        pass_past_len_1 = past_len
        pass_past_len_2 = ext.none_tensor

    ext_c.q_attn_forward_1(self.q_handle,
                            hidden_states,
                            batch_size,
                            q_len,
                            pass_past_len_1,
                            pass_past_len_2,
                            q_states,
                            k_states,
                            v_states,
                            constants.sin,
                            constants.cos,
                            pass_loras,
                            pass_lora_temp)

    # Shape for attention
    
    q_states = q_states.view(batch_size, q_len, num_attention_heads, head_dim)
    k_states = k_states.view(batch_size, q_len, num_key_value_heads, head_dim)
    v_states = v_states.view(batch_size, q_len, num_key_value_heads, head_dim)

    #Hacking chip stuff
    for chip_settings in settings:
        if chip_settings.q_in: q_states = hack_states(q_states, chip_settings.q_in, dim_info=flash_att_diminfo)
        if chip_settings.k_in: k_states = hack_states(k_states, chip_settings.k_in, dim_info=flash_att_diminfo)
        if chip_settings.v_in: v_states = hack_states(v_states, chip_settings.v_in, dim_info=flash_att_diminfo)
        if chip_settings.h_post: hidden_states = hack_states(hidden_states, chip_settings.h_post, dim_info=hidden_state_diminfo)
    # Regular (batched) attention with optional padding mask

    if cache is None or isinstance(cache, ExLlamaV2CacheBase):

        # Add keys and values to cache

        if cache is not None:
            if direct:
                k_states = batch_keys.narrow(0, 0, batch_size).narrow(1, 0, past_len + q_len)
                v_states = batch_values.narrow(0, 0, batch_size).narrow(1, 0, past_len + q_len)

            else:
                batch_keys, batch_values = cache.get_kv_state(self.layer_idx, batch_size, 0, past_len)
                new_keys = batch_keys.narrow(0, 0, batch_size).narrow(1, past_len, q_len)
                new_values = batch_values.narrow(0, 0, batch_size).narrow(1, past_len, q_len)
                new_keys.copy_(k_states)
                new_values.copy_(v_states)

                # Key/value tensors with past

                k_states = batch_keys.narrow(0, 0, batch_size).narrow(1, 0, past_len + q_len)
                v_states = batch_values.narrow(0, 0, batch_size).narrow(1, 0, past_len + q_len)

        # Torch matmul attention

        # TODO: To handle an exllamav2 update, I had to replace updated code with this, have to figure out hijacking it later

        # if self.ex_model.config.no_flash_attn or not has_flash_attn:
        #     attn_output = hacked_unflashed_attn_forward(self, attn_mask, settings, hack_states, num_key_value_groups, 
        #                      batch_size, head_dim, hidden_size, q_len, 
        #                      hidden_states, q_states, k_states, v_states)
        
        if self.model.config.no_flash_attn or not has_flash_attn or not attn_params.is_causal():

            q_states = q_states.transpose(1, 2)
            k_states = k_states.transpose(1, 2)
            v_states = v_states.transpose(1, 2)

            k_states = self.repeat_kv(k_states, num_key_value_groups)
            k_states = k_states.transpose(-1, -2)

            attn_weights = torch.matmul(q_states, k_states)
            k_states = None
            q_states = None

            attn_weights /= math.sqrt(head_dim)
            attn_mask = attn_params.get_attn_mask(hidden_states.device)
            if attn_mask is not None: attn_weights = attn_weights + attn_mask
            attn_weights = nn.functional.softmax(attn_weights, dim = -1, dtype = torch.float16)

            v_states = self.repeat_kv(v_states, num_key_value_groups)
            attn_output = torch.matmul(attn_weights, v_states)
            v_states = None

            attn_output = attn_output.transpose(1, 2)
            attn_output = attn_output.reshape((batch_size, q_len, hidden_size))
                
        # Flash Attention 2

        else:
        
        # TODO: To handle an exllamav2 update, I had to replace updated code with this, have to figure out hijacking it later
            
        #    attn_output = hacked_flash_attn_forward(settings, hack_states, 
        #                      batch_size, hidden_size, q_len, 
        #                      q_states, k_states, v_states)
        
            # TODO: Enable flash-attn with input mask
            attn_output = flash_attn_func(q_states, k_states, v_states, causal = True)
            attn_output = attn_output.reshape((batch_size, q_len, hidden_size))
            
        # xformers memory_efficient_attention

        # attn_output = xops.memory_efficient_attention(q_states, k_states, v_states, attn_bias = xops.LowerTriangularMask())
        # attn_output = attn_output.reshape((batch_size, q_len, hidden_size));

        # Torch SDP attention:

        # q_states = q_states.transpose(1, 2)
        # k_states = k_states.transpose(1, 2)
        # v_states = v_states.transpose(1, 2)
        #
        # # k_states = self.repeat_kv(k_states, num_key_value_groups)
        # # v_states = self.repeat_kv(v_states, num_key_value_groups)
        #
        # attn_output = F.scaled_dot_product_attention(q_states, k_states, v_states, attn_mask = attn_mask, is_causal = False)
        # attn_output = attn_output.transpose(1, 2)
        # attn_output = attn_output.reshape((batch_size, q_len, hidden_size))

        # Update 8-bit cache

        if cache is not None:
            cache.store_kv_state(self.layer_idx, batch_size, past_len, q_len)

    # Multiple caches

    else:
        assert attn_params.multi_cache
        attn_masks = attn_params.get_attn_masks(hidden_states.device)

        attn_outputs = []
        for i in range(len(cache)):

            # TODO: Once nested tensors are finalized in Torch, this could all be batched, probably

            # Add keys and values to cache

            batch_keys, batch_values = cache[i].get_kv_state(self.layer_idx, 1, 0, past_len[i])
            new_keys = batch_keys.narrow(1, past_len[i], q_len)
            new_values = batch_values.narrow(1, past_len[i], q_len)
            new_keys.copy_(k_states.narrow(0, i, 1))
            new_values.copy_(v_states.narrow(0, i, 1))

            # Store updated cache values

            cache[i].store_kv_state(self.layer_idx, 1, past_len[i], q_len)

            # Key/value tensors with past

            k_states_b = batch_keys.narrow(1, 0, past_len[i] + q_len)
            v_states_b = batch_values.narrow(1, 0, past_len[i] + q_len)

            # Torch matmul attention

            # TODO: enable flash-attn

            q_states_b = q_states.transpose(1, 2).narrow(0, i, 1)
            k_states_b = k_states_b.transpose(1, 2)
            v_states_b = v_states_b.transpose(1, 2)

            k_states_b = self.repeat_kv(k_states_b, num_key_value_groups)
            k_states_b = k_states_b.transpose(-1, -2)

            attn_weights = torch.matmul(q_states_b, k_states_b)
            q_states_b = None
            k_states_b = None

            attn_weights /= math.sqrt(head_dim)
            if attn_masks[i] is not None: attn_weights = attn_weights + attn_masks[i]
            attn_weights = nn.functional.softmax(attn_weights, dim = -1, dtype = torch.float16)

            v_states_b = self.repeat_kv(v_states_b, num_key_value_groups)
            attn_output_b = torch.matmul(attn_weights, v_states_b)
            v_states_b = None

            attn_outputs.append(attn_output_b)
                
        q_states = None
        k_states = None
        v_states = None

        attn_output = torch.cat(attn_outputs, dim = 0)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape((batch_size, q_len, hidden_size))

   
    
    #Hacking chip stuff
    for chip_settings in settings:
        # Due to this being a loop now, I need to make sure the input and output are the same variable
        # I'm not sure if this should both be attn_output or both hidden_states
        # My instinct is to have the line below pass attn_output into hack_states, and not using hidden_states here
        if chip_settings.a_c: attn_output = hack_states(hidden_states, chip_settings.a_c)
    
    #Output projection
    ext_c.q_attn_forward_2(self.q_handle,
                            hidden_states,
                            attn_output,
                            batch_size,
                            q_len,
                            pass_loras,
                            pass_lora_temp)
    for chip_settings in settings:
        # attn_output gets set to None right below this and then hidden_states are returned
        # Should this deal with hidden_states?
        if chip_settings.a_po: attn_output = hack_states(attn_output, chip_settings.a_po)

    attn_output = None
    attn_weights = None

    return hidden_states


def hacked_flash_attn_forward(hack_settings, hack_states, 
                             batch_size, hidden_size, q_len, 
                             q_states, k_states, v_states):
     #Hacking chip stuff
    for chip_settings in hack_settings:
        if chip_settings.k_all: k_states = hack_states(k_states, chip_settings.k_all, flash_att_diminfo) 
        if chip_settings.v_all: v_states = hack_states(v_states, chip_settings.v_all, flash_att_diminfo) 
    
    attn_output = flash_attn_func(q_states, k_states, v_states, causal = True)
    attn_output = attn_output.reshape((batch_size, q_len, hidden_size))
    return attn_output

def hacked_unflashed_attn_forward(self, attn_mask, hack_settings, hack_states, num_key_value_groups, 
                             batch_size, head_dim, hidden_size, q_len, 
                             hidden_states, q_states, k_states, v_states):
    q_states = q_states.transpose(1, 2)
    k_states = k_states.transpose(1, 2)
    v_states = v_states.transpose(1, 2)            

    k_states = self.repeat_kv(k_states, num_key_value_groups)    
    v_states = self.repeat_kv(v_states, num_key_value_groups)
    #Hacking chip stuff
    for chip_settings in hack_settings:
        if chip_settings.k_all: k_states = hack_states(k_states, chip_settings.k_all, dim_info=unflash_att_diminfo)
        if chip_settings.v_all: v_states = hack_states(v_states, chip_settings.v_all, dim_info=unflash_att_diminfo)

    k_states = k_states.transpose(-1, -2)
    attn_weights = torch.matmul(q_states, k_states)
    k_states = None
    q_states = None

    attn_weights /= math.sqrt(head_dim)
    if attn_mask is not None: attn_weights = attn_weights + attn_mask
    attn_weights = nn.functional.softmax(attn_weights, dim = -1, dtype = torch.float16)

    
    attn_output = torch.matmul(attn_weights, v_states)
    
    for chip_settings in hack_settings:
        if chip_settings.a_ho: attn_output = hack_states(attn_output, chip_settings.a_ho, dim_info=unflash_att_diminfo)
        
    v_states = None

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape((batch_size, q_len, hidden_size))
    return attn_output
    
def multi_cache_attn_forward(self, cache, attn_mask, hack_settings, hack_states, num_key_value_groups, 
                             batch_size, head_dim, past_len, q_len, 
                             hidden_states, q_states, k_states, v_states):
    attn_outputs = []
    for i in range(len(cache)):
        # TODO: Once nested tensors are finalized in Torch, this could all be batched, probably

        # Add keys and values to cache

        batch_keys, batch_values = cache[i].get_kv_state(self.layer_idx, batch_size, 0, past_len)
        new_keys = batch_keys.narrow(1, past_len[1][i], q_len)
        new_values = batch_values.narrow(1, past_len[1][i], q_len)
        new_keys.copy_(k_states.narrow(0, i, 1))
        new_values.copy_(v_states.narrow(0, i, 1))

        # Key/value tensors with past

        k_states_b = batch_keys.narrow(1, 0, past_len[1][i] + q_len)
        v_states_b = batch_values.narrow(1, 0, past_len[1][i] + q_len)

        # Torch matmul attention

        # TODO: enable flash-attn

        q_states_b = q_states.transpose(1, 2).narrow(0, i, 1)
        k_states_b = k_states_b.transpose(1, 2)
        v_states_b = v_states_b.transpose(1, 2)

        k_states_b = self.repeat_kv(k_states_b, num_key_value_groups)
        k_states_b = k_states_b.transpose(-1, -2)
        
        #Hacking chip stuff
        for chip_settings in hack_settings:
            if chip_settings.q1: q_states = hack_states(q_states, chip_settings.q1)
            if chip_settings.k1: k_states = hack_states(k_states, chip_settings.k1)
            if chip_settings.v1: v_states = hack_states(v_states, chip_settings.v1)

        attn_weights = torch.matmul(q_states_b, k_states_b)
        q_states_b = None
        k_states_b = None

        attn_weights /= math.sqrt(head_dim)
        if attn_mask is not None: attn_weights = attn_weights + attn_mask[i]
        attn_weights = nn.functional.softmax(attn_weights, dim = -1, dtype = torch.float16)

        v_states_b = self.repeat_kv(v_states_b, num_key_value_groups)
        attn_output_b = torch.matmul(attn_weights, v_states_b)
        v_states_b = None
        
        for chip_settings in hack_settings:
            if chip_settings.a_ho: attn_output_b = hack_states(attn_output_b, chip_settings.a_ho)
        attn_outputs.append(attn_output_b)
    return attn_outputs
    
