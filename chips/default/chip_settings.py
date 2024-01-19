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
    
    chip.logits_settings = LayerSettings(weight=weight, cfg_func=cfg_default)

    return chip
