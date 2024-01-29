import torch
from extensions.BrainHackingChip.settings_classes import LayerSettings, AttnSettings, VectorSettings, Value
from .chip_ui import *

from .chip_prompt import gen_full_prompt as gen_full_prompt2

def gen_full_prompt(user_input, state, **kwargs): # This call finds the [[POSITVE]] and [[NEGATIVE]] tags and splits the prompts
    baseprompt, prompts = gen_full_prompt2(user_input, state, **kwargs) # I'm sure there's a better way to do this, oh well
    return baseprompt, prompts

def brainhackingchip_settings(chip, params, last_kv_layer, head_layer):
    weight = ui_params['weight']['attributes']['value']
    
    def cfg_default(tensor, settings, hackingchip):
      if hackingchip.prompts.numneg > 0:
        x_neg_steering = tensor[hackingchip.prompts.numpos:hackingchip.prompts.negend]
        x_neg_steering = torch.mean(x_neg_steering, dim=0, keepdim=False) # probably not the best way to handle this but oh well
        x_neg_steering = settings.weight * (x_neg_steering - tensor[0])

        # It's important to steer all of the vectors, or else the difference artificially accumulates and accelerates.
        tensor -= x_neg_steering
      return tensor
    
    chip.layer_settings[last_kv_layer - 1] = LayerSettings(weight=weight, cfg_func=cfg_default)
    chip.layer_settings[last_kv_layer + 1] = LayerSettings(weight=weight, cfg_func=cfg_default)

    # Logits only
    # chip.logits_settings = LayerSettings(weight=weight, cfg_func=cfg_default)



    # CFG specifically for attention layer vectors
    # Similar to DRµGS, using the H, Q, K, V, A vectors: https://github.com/EGjoni/DRUGS/blob/main/porting/A%20Guide%20to%20Making%20DRUGS.md
    
    # You can do custom cfg_func with Q, K, V as well! It's the exact same function signature (so can use the same function for all if you want)
    
    # Uncomment for attention
    # attn_weight = weight
    # attn_cfg_func = cfg_default
    # attn_test = AttnSettings()
    
    # Uncomment any of the H, Q, K, V, A lines below (can be used together), also need to set chip.attn_settings so uncommenting here won't do anything alone
    
    # attn_test.h = VectorSettings(weight=attn_weight, cfg_func=attn_cfg_func)
    # attn_test.q_in = VectorSettings(weight=attn_weight, cfg_func=attn_cfg_func)
    # attn_test.k_in = VectorSettings(weight=attn_weight, cfg_func=attn_cfg_func)
    # attn_test.v_in = VectorSettings(weight=attn_weight, cfg_func=attn_cfg_func)
    # attn_test.a_po = VectorSettings(weight=attn_weight, cfg_func=attn_cfg_func)
    
    # Uncomment this to use attn_test
    # chip.attn_settings = [attn_test] * len(chip.attn_settings) # testing every attention layer
    
    
        
    return chip
