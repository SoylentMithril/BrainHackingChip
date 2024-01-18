import torch
from extensions.BrainHackingChip.settings_classes import LayerSettings, AttnSettings, VectorSettings, Value
from .chip_ui import *

from .chip_prompt import gen_full_prompt as gen_full_prompt2

def gen_full_prompt(user_input, state, **kwargs): # This call finds the [[POSITVE]] and [[NEGATIVE]] tags and splits the prompts
    baseprompt, prompts = gen_full_prompt2(user_input, state, **kwargs) # I'm sure there's a better way to do this, oh well
    return baseprompt, prompts

def brainhackingchip_settings(chip, params, last_kv_layer, head_layer):
    """
    This will be rewritten, before I was talking about each layer individually (attention, feed forward, etc)
    Moving toward an attention layer focused process with H, Q, K, V, A similar to DRµGS
    Support for H, Q, K, V, A below, after this initial section for individual layers that I'm still supporting cause it has a good CFG setting
    """
    
    """
    Layer count is head_layer + 1
    layer_settings indices range from 0 to head_layer
    layer_settings[last_kv_layer - 1] is the last layer that will influence the preprocessed cache (because of the order this is done)
    layer_settings[head_layer] is the last index and will output logits rather than a hidden state
    
    How the CFG works:
    1. The difference between the tensors for negative prompt - positive prompt is found
    2. That difference is multiplied by the weight in LayerSettings
    3. That difference is subtracted from the tensors (steering positive away from negative by their difference)
    The larger the weight, the more intense of an effect
    
    
    
    There is also support for a custom CFG function instead of the default above, by setting cfg_func in any layer you want to override
    LayerSettings(cfg_func=YourFuncionHere)
    The function will be called with these parameters: new_tensors = cfg_func(tensor, settings, hackingchip)
      Tensor will be the full block of positive and negative tensors
      Settings will be the respective settings object (layer or QKV settings)
      Hackingchip is included for extra info, such as hackingchip.prompt.numpos, hackingchip.prompt.numneg, and hackingchip.prompt.negend
    You must return the new full tensors (same shape as input tensors) after your modification
    
    For an example of a custom cfg_func, see cfg_repulsor below
    """
    
    # weight = params['weight']['value'] if 'weight' in params and 'value' in params['weight'] else 0.2 # still no autoload so have to be careful here
    weight = ui_params['weight']['attributes']['value']
    
    def cfg_default(tensor, settings, hackingchip):
      if hackingchip.prompts.numneg > 0:
        x_neg_steering = tensor[hackingchip.prompts.numpos:hackingchip.prompts.negend]
        x_neg_steering = torch.mean(x_neg_steering, dim=0, keepdim=False) # probably not the best way to handle this but oh well
        x_neg_steering = settings.weight * (x_neg_steering - tensor[0])

        # It's important to steer all of the vectors, or else the difference artificially accumulates and accelerates.
        tensor -= x_neg_steering
        
      return tensor
    
        
    # Set a cfg_func for my thought CFG example
    thought_cfg_func = cfg_default
    
    # The weight of the CFG for thoughts in this example, change this value to change how strongly thoughts are affected by negative prompts
    # The amount of weight to use seems to depend on how many layers you are putting it on
    # It seems like once you accumulate 0.5 weight among all layers or more, things can get weird. The default puts 0.2 weight into two different layers.
    thought_weight = weight
    
    chip.layer_settings[last_kv_layer - 1] = LayerSettings(weight=thought_weight, cfg_func=thought_cfg_func)
    chip.layer_settings[last_kv_layer + 1] = LayerSettings(weight=thought_weight, cfg_func=thought_cfg_func)
    
    

    # The head_layer is the final layer that outputs values for all of the logits (every possible token)
    # This will do "normal" CFG that directly affects token probabilities
    # This can help the thought CFG's impact get amplified enough to change the output reliably, but also tends to make things crazier
    # Also, if the LLM is cutting off output prematurely, this may help with that
    
    # Currently commented out below, but can be enabled
    
    # logits_weight = 0.1
    # chip.layer_settings[head_layer] = LayerSettings(weight=logits_weight)
    

    
    
    
    
    
    # CFG specifically for attention layer vectors, this will likely be what future BHC development centers on
    # Similar to DRµGS, using the H, Q, K, V, A vectors: https://github.com/EGjoni/DRUGS/blob/main/porting/A%20Guide%20to%20Making%20DRUGS.md
    
    attn_weight = weight
    
    # You can do custom cfg_func with Q, K, V as well! It's the exact same function signature (so can use the same function for all if you want)
    attn_cfg_func = cfg_default
    
    attn_test = AttnSettings()
    
    # Uncomment any of the H, Q, K, V, A lines below (can be used together), also need to set chip.attn_settings so uncommenting here won't do anything alone
    
    # attn_test.h = VectorSettings(weight=attn_weight, cfg_func=attn_cfg_func)
    # attn_test.q = VectorSettings(weight=attn_weight, cfg_func=attn_cfg_func)
    # attn_test.k = VectorSettings(weight=attn_weight, cfg_func=attn_cfg_func)
    # attn_test.v = VectorSettings(weight=attn_weight, cfg_func=attn_cfg_func)
    # attn_test.a = VectorSettings(weight=attn_weight, cfg_func=attn_cfg_func)
    
    # Uncomment one line below, first line uses every single attention layer and 2nd line only uses the very last attention layer
    
    # chip.attn_settings = [attn_test] * len(chip.attn_settings) # testing every attention layer
    # chip.attn_settings[-1] = attn_test # testing only the last attention
    
    
        
    return chip
