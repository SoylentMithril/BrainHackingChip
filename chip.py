import copy
import importlib

from modules import shared, chat

import torch
import random

from modules.exllamav2 import Exllamav2Model


import math
from torch import nn
from .util.general_stuff import *

from extensions.BrainHackingChip.settings_classes import HackingchipSettings
from extensions.BrainHackingChip.hackingchip_classes import Hackingchip, HackingchipPrompts

# Override functions to inject hackingchip behavior into model loaders. These functions need to be kept up to date with oobabooga's exllamav2

# Here is the actual construction and injection of the hackingchip into the model

def gen_full_prompt(user_settings, ui_settings, ui_params, user_input, state, **kwargs):
    def default_gen_full_prompt(user_input, state, **kwargs):
        baseprompt = chat.generate_chat_prompt(user_input, state, **kwargs)
        prompts = HackingchipPrompts([baseprompt], 1, 0)
        return baseprompt, prompts
    
    settings = None
    
    if ui_settings['on']:
        loader_module_path = None
        if shared.model != None:
            if isinstance(shared.model, Exllamav2Model):
                loader_module_path = "exllamav2"
                
        if loader_module_path:
            loader_module = importlib.import_module("extensions.BrainHackingChip.{loader}".format(loader=loader_module_path))
            importlib.reload(loader_module)
            
            baseprompt = None
            for user_settings_single in user_settings:
                if hasattr(user_settings_single, 'gen_full_prompt'): # Prompt generation override for multiple batches, maybe other things?
                    baseprompt, prompts = user_settings_single.gen_full_prompt(user_input, state, **kwargs)
                    break  # only picking the first one for now, later one will add a way to choose which parses the prompt
            if not baseprompt:
                baseprompt, prompts = default_gen_full_prompt(user_input, state, **kwargs) # prepare hackingchip prompts
            
            model_info = loader_module.get_model_info()
            
            temp_ui_params = {} # Need to adjust to GUI changes, disabling ui_params and doing this workaround for now
            settings = [user_settings_single.brainhackingchip_settings(
                            HackingchipSettings(model_info['layers_count'], model_info['attn_layers']),
                            temp_ui_params, model_info['last_kv_layer'], model_info['head_layer']
                        ) for user_settings_single in user_settings]            
            # settings = [user_settings_single.brainhackingchip_settings(
            #                 HackingchipSettings(model_info['layers_count'], model_info['attn_layers']),
            #                 ui_params_single, model_info['last_kv_layer'], model_info['head_layer']
            #             ) for user_settings_single, ui_params_single in zip(user_settings, ui_params)]            
            # settings is an array now, can support multichip
            hackingchip = Hackingchip(ui_settings, settings, prompts, len(model_info['attn_layers']))
            
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
    else: # Just pass through to default behavior
        return chat.generate_chat_prompt(user_input, state, **kwargs)
                        
                    