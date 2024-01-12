import copy
import re
import importlib

from modules import shared, chat

from modules.text_generation import get_encoded_length, get_max_prompt_length
from modules.extensions import apply_extensions
from modules.chat import get_generation_prompt

import torch
import random

from modules.exllamav2 import Exllamav2Model

from exllamav2.generator import ExLlamaV2Sampler, ExLlamaV2StreamingGenerator
from exllamav2.cache import ExLlamaV2CacheBase
from exllamav2.model import _torch_device, ExLlamaV2
from exllamav2.compat import safe_move_tensor
from exllamav2 import (
    ExLlamaV2Cache,
    ExLlamaV2Cache_8bit,
)
from exllamav2.attn import ExLlamaV2Attention

from jinja2.sandbox import ImmutableSandboxedEnvironment
jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
from functools import partial

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

from extensions.BrainHackingChip.settings_classes import HackingchipSettings

# Override functions to inject hackingchip behavior into model loaders. These functions need to be kept up to date with oobabooga's exllamav2

# Here is the actual construction and injection of the hackingchip into the model

class Hackingchip:
    def __init__(self, ui_settings, settings, prompts):
        self.ui_settings = ui_settings
        self.settings = settings
        self.prompts = prompts
        
class HackingchipPrompts:
    def __init__(self, prompts, numpos, numneg):
        self.batch_prompts = prompts
        self.numpos = numpos
        self.numneg = numneg
        self.negend = numpos + numneg
        self.batch_size = numpos + numneg
        
def gen_full_prompt(user_settings, ui_settings, ui_params, user_input, state, **kwargs):
    settings = None
    
    if ui_settings['on']:
        loader_module_path = None
        if shared.model != None:
            if isinstance(shared.model, Exllamav2Model):
                loader_module_path = "exllamav2"
                
        if loader_module_path:
            loader_module = importlib.import_module("extensions.BrainHackingChip.{loader}".format(loader=loader_module_path))
            importlib.reload(loader_module)
            
            baseprompt, prompts = gen_full_prompt2(user_input, state, **kwargs) # prepare hackingchip prompts
            
            model_info = loader_module.get_model_info()
            
            settings = user_settings.brainhackingchip_settings(HackingchipSettings(model_info['layers_count'], model_info['attn_layers']),
                                                               ui_params, model_info['last_kv_layer'], model_info['head_layer']) # prepare hackingchip settings
            
            hackingchip = Hackingchip(ui_settings, settings, prompts)
            
            loader_module.hijack_loader(hackingchip)
           
        if ui_settings['output_prompts']:
            print("Hackingchip prompts:")
            for prompt in hackingchip.prompts.batch_prompts:
                print(prompt)
        
        return baseprompt
    else:
        # Should I warn the user that they aren't able to use hackingchip with their current model loader? Or would that be annoying?
        if settings is None: print("Unsupported model loader: Brain-Hacking Chip won't work with it")
        return chat.generate_chat_prompt(user_input, state, **kwargs)
            
            
            
# I wrote the below code awhile ago as a quick hack to get things working then shoddily reworked it multiple times, please forgive me for my sins

# The code below prepares the prompts using the [[POSITIVE]] and [[NEGATIVE]] tags
    
def gen_full_prompt2(user_input, state, **kwargs): # modifies hackingchip and state
    # global last_prompt
    # global last_chip
    
    prompt = None
    
    # custom prompt generation stuff could go here

    # if prompt is not None and kwargs.get('_continue', False):
    #     prompt = last_prompt
    #     hackingchip = last_chip
        
    numpos = 1
    numneg = 0
                
    if prompt is None:
        positive_context, negative_context, positive_context_extras, negative_context_extras = process_context(state['context'])
        positive_context_instruct, negative_context_instruct, positive_context_instruct_extras, negative_context_instruct_extras = process_context(state['custom_system_message'])
        
        if len(negative_context) > 0 and len(negative_context_instruct) == 0:
            negative_context_instruct = positive_context_instruct
        
        positive_extras = {}
        negative_extras = {}
        
        for name, text in positive_context_extras.items():
            if name not in positive_extras:
                positive_extras[name] = ExtraInfo()
                positive_extras[name].inst = positive_context_instruct
            positive_extras[name].char = text
            
        for name, text in positive_context_instruct_extras.items():
            if name not in positive_extras:
                positive_extras[name] = ExtraInfo()
                positive_extras[name].char = positive_context
            positive_extras[name].inst = text
            
        for name, text in negative_context_extras.items():
            if name not in negative_extras:
                negative_extras[name] = ExtraInfo()
                negative_extras[name].inst = negative_context_instruct
            negative_extras[name].char = text
            
        for name, text in negative_context_instruct_extras.items():
            if name not in negative_extras:
                negative_extras[name] = ExtraInfo()
                negative_extras[name].char = negative_context
            negative_extras[name].inst = text
            
        state['context'] = positive_context
        state['custom_system_message'] = positive_context_instruct
        posprompt = generate_chat_prompt(user_input, state, **kwargs)
        prompt = [posprompt]

        if positive_extras:
            for name, extras in positive_extras.items():
                state['context'] = extras.char
                state['custom_system_message'] = extras.inst
                prompt.append(generate_chat_prompt(user_input, state, **kwargs))
                numpos += 1

        if negative_extras:
            for name, extras in negative_extras.items():
                state['context'] = extras.char
                state['custom_system_message'] = extras.inst
                prompt.append(generate_chat_prompt(user_input, state, **kwargs))
                numneg += 1
        
        if len(negative_context) + len(negative_context_instruct) > 0:
            state['context'] = negative_context
            state['custom_system_message'] = negative_context_instruct
            prompt.append(generate_chat_prompt(user_input, state, **kwargs))
            numneg += 1
            
        state['context'] = positive_context
        state['custom_system_message'] = positive_context_instruct
        
        prompt_info = HackingchipPrompts(prompt, numpos, numneg)
        
        # TODO: load the default negative cfg here in state for convenience
        
        prompt = posprompt # for compatibility
    else:
        positive, negative, positive_extras, negative_extras = process_context(prompt)

        prompt = [positive]
        
        if positive_extras:
            prompt.append(positive_extras)
            numpos += positive_extras.len()

        if len(negative) > 0:
            prompt.append(negative)
            numneg += 1
            
        if negative_extras:
            prompt.append(negative_extras)
            numneg += negative_extras.len()

        prompt_info = HackingchipPrompts(prompt, numpos, numneg)
        
        prompt = positive
        
    # If the user hits continue, this is how I can continue without breaking things... although I haven't completed this currently
    # Actually, this is deprecated now, commented out all related code and will probably remove
    # last_prompt = prompt
    # last_chip = copy.deepcopy(hackingchip)
    
    # TODO <|nochat|> may be bad for continue as is, because it may also exclude the AI's output so far
    # I'm sure it can be fixed if it's busted, need to look into it
        
    for oneprompt in prompt:
        if '<|nochat|>' in oneprompt:
            oneprompt = oneprompt.replace('<|nochat|>', '').replace('<|chat|>', '')
            
    return prompt, prompt_info
            
class ExtraInfo:
    def __init__(self):
        self.char = ''
        self.inst = ''
        
def process_context(context):
    pattern = r"(?:\[\[(?P<name>[^\]]+)\]\]\n)?(?P<text>(?:(?!\n\[\[[^\]]+\]\]\n).)*)"
    
    if not re.match(r'^\[\[(.*?)\]\]\n', context): context = "[[SHARED]]\n" + context
    
    matches = re.finditer(pattern, context, re.DOTALL)
    
    regions = {}
    
    positive = ''
    negative = ''
    
    positive_extras = {}
    negative_extras = {}

    for match in matches:
        if match.group("name") is not None:
            name = match.group("name").upper().strip()
            text = match.group("text") # don't strip
            
            if name.startswith('POSITIVE') and name != 'POSITIVE':
                positive_extras[name] = text
            if name.startswith('NEGATIVE') and name != 'NEGATIVE':
                negative_extras[name] = text
            
            regions[name] = text
        
    if 'POSITIVE' in regions:
        positive = regions['POSITIVE']
    elif positive_extras:
        name, text = positive_extras.popitem()
        positive = text
    else:
        if 'SHARED' in regions:
            positive = regions['SHARED']
        else:
            positive = context # maybe not good?
    
    if 'NEGATIVE' in regions:
        negative = regions['NEGATIVE']
    elif negative_extras:
        name, text = negative_extras.popitem()
        negative = text
    
    for name, text in regions.items():
        positive = positive.replace('{{' + name + '}}', text)
        negative = negative.replace('{{' + name + '}}', text)
        for name2, text2 in positive_extras.items():
            positive_extras[name2] = text2.replace('{{' + name + '}}', text)
        for name2, text2 in negative_extras.items():
            negative_extras[name2] = text2.replace('{{' + name + '}}', text)
       
    return positive, negative, positive_extras, negative_extras
        
# Just copying the entirety of generate_chat_prompt so I can put <|nochat|> support in it

def generate_chat_prompt(user_input, state, **kwargs):
    impersonate = kwargs.get('impersonate', False)
    _continue = kwargs.get('_continue', False)
    also_return_rows = kwargs.get('also_return_rows', False)
    history = kwargs.get('history', state['history'])['internal']

    # Templates
    chat_template = jinja_env.from_string(state['chat_template_str'])
    instruction_template = jinja_env.from_string(state['instruction_template_str'])
    chat_renderer = partial(chat_template.render, add_generation_prompt=False, name1=state['name1'], name2=state['name2'])
    instruct_renderer = partial(instruction_template.render, add_generation_prompt=False)

    messages = []

    if state['mode'] == 'instruct':
        renderer = instruct_renderer
        if state['custom_system_message'].strip() != '':
            messages.append({"role": "system", "content": state['custom_system_message']})
    else:
        renderer = chat_renderer
        if state['context'].strip() != '':
            messages.append({"role": "system", "content": state['context']})

    insert_pos = len(messages)
    for user_msg, assistant_msg in reversed(history):
        user_msg = user_msg.strip()
        assistant_msg = assistant_msg.strip()

        if assistant_msg:
            messages.insert(insert_pos, {"role": "assistant", "content": assistant_msg})

        if user_msg not in ['', '<|BEGIN-VISIBLE-CHAT|>']:
            messages.insert(insert_pos, {"role": "user", "content": user_msg})

    user_input = user_input.strip()
    if user_input and not impersonate and not _continue:
        messages.append({"role": "user", "content": user_input})

    def make_prompt(messages):
        if state['mode'] == 'chat-instruct' and _continue:
            prompt = renderer(messages=messages[:-1])
        else:
            prompt = renderer(messages=messages)

        if state['mode'] == 'chat-instruct':
            outer_messages = []
            if state['custom_system_message'].strip() != '':
                outer_messages.append({"role": "system", "content": state['custom_system_message']})

            command = state['chat-instruct_command']
            command = command.replace('<|character|>', state['name2'] if not impersonate else state['name1'])
            command = command.replace('<|prompt|>', prompt)

            if _continue:
                prefix = get_generation_prompt(renderer, impersonate=impersonate, strip_trailing_spaces=False)[0]
                prefix += messages[-1]["content"]
            else:
                prefix = get_generation_prompt(renderer, impersonate=impersonate)[0]
                if not impersonate:
                    prefix = apply_extensions('bot_prefix', prefix, state)
                    
            if '<|nochat|>' not in user_input:
                outer_messages.append({"role": "user", "content": command})
                outer_messages.append({"role": "assistant", "content": prefix})

            prompt = instruction_template.render(messages=outer_messages)
            suffix = get_generation_prompt(instruct_renderer, impersonate=False)[1]
            prompt = prompt[:-len(suffix)]

        else:
            if _continue:
                suffix = get_generation_prompt(renderer, impersonate=impersonate)[1]
                prompt = prompt[:-len(suffix)]
            else:
                prefix = get_generation_prompt(renderer, impersonate=impersonate)[0]
                if state['mode'] == 'chat' and not impersonate:
                    prefix = apply_extensions('bot_prefix', prefix, state)

                prompt += prefix

        return prompt

    prompt = make_prompt(messages)

    # Handle truncation
    max_length = get_max_prompt_length(state)
    while len(messages) > 0 and get_encoded_length(prompt) > max_length:
        # Try to save the system message
        if len(messages) > 1 and messages[0]['role'] == 'system':
            messages.pop(1)
        else:
            messages.pop(0)

        prompt = make_prompt(messages)

    if also_return_rows:
        return prompt, [message['content'] for message in messages]
    else:
        return prompt
                    