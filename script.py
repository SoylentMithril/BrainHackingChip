import importlib
import gradio as gr
from modules import shared

from modules.ui import create_refresh_button

import os
import fnmatch

params = {
    'is_tab': True,
    'display_name': 'Brain-Hacking Chip'
}

gradio = {}

max_sliders = 10 # This will probably have to be higher
chip_path = "extensions.BrainHackingChip.chips.{name}.chip_settings"

# Globals
chip_settings = None
# This should probably just be a dictionary, but I'm not sure if that would mess up gradio
slider_map = [] # store dictionary key of each slider in order
sliders = []
slider_values = []

ui_settings = {
    'on': True,
    'output_prompts': False,
    'sample_other_prompts': False
    # 'output_extra_samples': False
}

def get_available_files():
    directory_to_search = './extensions/BrainHackingChip/chips/'
    file_to_find = 'chip_settings.py'
    
    subdirectories = ['default']
    
    for root, dirs, files in os.walk(directory_to_search):
        if file_to_find in files:
            subdirectories.append(os.path.basename(root))

    return subdirectories

def select_file(filename):
    global chip_settings
    
    def do_default():
        try:
            global chip_settings
            chip_settings = importlib.import_module("extensions.BrainHackingChip.chip_settings") # default settings
        except Exception as e:
            print("This shouldn't happen")
    
    if filename == "default":
        do_default()
    else:
        path = chip_path.format(name=filename)
        
        try:
            chip_settings = importlib.import_module(path)
        except Exception as e:
            do_default()
        
    importlib.reload(chip_settings)
    
    return populate_sliders()

def populate_base_sliders():
    global slider_values, slider_map, sliders
    mu = shared.args.multi_user
    
    slider_values = [0] * len(params)
    slider_map = [] # reset slider_map
    sliders = [] # reset sliders
    count = 0

    if chip_settings and hasattr(chip_settings, 'params'):
        for key, value in chip_settings.params.items():
            slider_values[count] = value.start
            
            slider_map.append(key)
            sliders.append(gr.Slider(visible=True, 
                                     interactive=not mu,
                                     minimum=value.min if hasattr(value, 'min') else 0.0,
                                     maximum=value.max if hasattr(value, 'max') else 0.0,
                                     value=value.start if hasattr(value, 'start') else 0.0,
                                     step=value.step if hasattr(value, 'step') else 0.0,
                                     label=value.name if hasattr(value, 'name') else '',
                                     info=value.description if hasattr(value, 'description') else None
                                     ))
            
            count += 1
            if count >= max_sliders:
                count = max_sliders
                break

def populate_sliders():
    global chip_settings, slider_values, slider_map, sliders
    mu = shared.args.multi_user
    
    slider_map = [] # reset slider_map
    sliders = [] # reset sliders
    count = 0

    if chip_settings and hasattr(chip_settings, 'ui_params'):
        slider_values = [0] * len(chip_settings.ui_params)
        
        for key, value in chip_settings.ui_params.items():
            slider_values[count] = value.start
            
            slider_map.append(key)
            sliders.append(gr.Slider(visible=True, 
                                     interactive=not mu,
                                     minimum=value.min if hasattr(value, 'min') else 0.0,
                                     maximum=value.max if hasattr(value, 'max') else 0.0,
                                     value=value.start if hasattr(value, 'start') else 0.0,
                                     step=value.step if hasattr(value, 'step') else 0.0,
                                     label=value.name if hasattr(value, 'name') else '',
                                     info=value.description if hasattr(value, 'description') else None
                                     ))
            
            count += 1
            if count >= max_sliders:
                count = max_sliders
                break
            
    return sliders + [gr.Slider(visible=False)]*(max_sliders - len(sliders))

def get_slider_values():
    global slider_values, slider_map
    
    values = {}
    
    for index, value in enumerate(slider_values):
        values[slider_map[index]] = value
        
    return values

def make_slider_change(index):
    def slider_change(slider):
        global slider_values
        slider_values[index] = slider
        # print(str(index) + ": " + str(slider))
        
    return slider_change

def on_switch_change(value):
    ui_settings['on'] = value
    
def output_prompts_change(value):
    ui_settings['output_prompts'] = value
    
def sample_other_prompts_change(value):
    ui_settings['sample_other_prompts'] = value
        
# I'm learning gradio with this function, bear with me here
def ui():
    global sliders, gradio
    mu = shared.args.multi_user
    
    with gr.Row():
        with gr.Column():
            with gr.Row():
                gradio['on_switch'] = gr.Checkbox(label="Activate Brain-Hacking Chip", value=True)
            with gr.Row():
                gradio['output_prompts'] = gr.Checkbox(label="Debug: Output Prompts", value=False, info='Print all prompts to the console.')
                
                # This isn't working now and I'm not sure why, made it invisible for now
                gradio['sample_other_prompts'] = gr.Checkbox(label="Debug: Sample Other Prompts", value=False, info='Samples tokens from any extra prompts and prints their output to the console.', visible=False)
            sliders_full = []
            for i in range(max_sliders):
                with gr.Row():
                    t = gr.Slider(0.0, 1.0, value=0.0, step=0.1, visible=False, interactive=not mu)
                    sliders_full.append(t)
                
        with gr.Column():
            with gr.Row():
                gradio['file_select'] = gr.Dropdown(choices=get_available_files(), value=None, label='Settings File', elem_classes='slim-dropdown', interactive=not mu)  
                create_refresh_button(gradio['file_select'], lambda: None, lambda: {'choices': get_available_files()}, 'refresh-button', interactive=not mu)
            with gr.Row():
                # Currently not using this, making it invisible for now... Not sure if I want script editing in here or not
                gradio['file_text'] = gr.Textbox(value='', lines=27, elem_id='textbox-notebook', elem_classes=['textbox', 'add_scrollbar'], visible=False)
                
    gradio['file_select'].change(select_file, gradio['file_select'], sliders_full)
    
    # This doesn't trigger the change event... Why? It sets the value in the GUI, but I can't trigger the change event
    # gradio['file_select'].value = "default"
    
    gradio['on_switch'].change(on_switch_change, gradio['on_switch'])
    gradio['output_prompts'].change(output_prompts_change, gradio['output_prompts'])
    gradio['sample_other_prompts'].change(sample_other_prompts_change, gradio['sample_other_prompts'])
    
    # I wasn't sure how to get up to date slider values for every generation, so I just had to add change event listeners to them all
    # I know it's inefficient, I know it's bad, but it's all I've got right now
    for index, slider in enumerate(sliders_full):
        slider.input(make_slider_change(index), slider)
    
def custom_generate_chat_prompt(user_input, state, **kwargs):
    global ui_settings, chip_settings
    ui_params = get_slider_values()
    
    chip = importlib.import_module("extensions.BrainHackingChip.chip")
    importlib.reload(chip)
    
    if not chip_settings: # Just in case
        chip_settings = importlib.import_module("extensions.BrainHackingChip.chip_settings")
        importlib.reload(chip_settings)
        
    prompt = chip.gen_full_prompt(chip_settings, ui_settings, ui_params, user_input, state, **kwargs)
    
    return prompt
