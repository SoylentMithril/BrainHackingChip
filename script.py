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
default_max_widgets = 10 # This will probably have to be higher
default_max_widgets = 10
chip_path = "extensions.BrainHackingChip.chips.{name}.chip_settings"
chip_ui_path = "extensions.BrainHackingChip.chips.{name}.chip_ui"
active_chip_path = "extensions.BrainHackingChip.chips.default.chip_settings"
active_chip_ui_path = "extensions.BrainHackingChip.chips.default.chip_ui"
chip_blocks = {}
chipblocks_list = []
# Globals
chip_settings = None
# This should probably just be a dictionary, but I'm not sure if that would mess up gradio
widget_keys = [] # store dictionary key of each slider in order
widgets = []
widget_values = []

ui_settings = {
    'on': True,
    'output_prompts': False,
    'sample_other_prompts': False
    # 'output_extra_samples': False
}

def get_available_files():
    directory_to_search = './extensions/BrainHackingChip/chips/'
    file_to_find = 'chip_settings.py'
    
    subdirectories = []
    
    try:
        for root, dirs, files in os.walk(os.path.join(os.getcwd(), directory_to_search)):
            if file_to_find in files:
                subdirectories.append(os.path.basename(root))
    except Exception as e:
        print("Error loading chips")

    return subdirectories

def make_chip_blocks(mu):
    global chip_blocks, chipblocks_list, dict_all_chip_widgets
    dict_all_chip_widgets = {}
    chipblocks_list = []
    chip_blocks = {}
    file_list = get_available_files()
    all_chip_widgets = []
    try:        
        for filename in file_list:
            path = chip_ui_path.format(name=filename)
            temp_chip_ui = importlib.import_module(path)
            with gr.Group(visible=False) as widg_block:
                chip_widg_elems, widget_keys, widget_values, ui_components = populate_widgets(temp_chip_ui.ui_params, mu, filename)
                all_chip_widgets = [*all_chip_widgets, *chip_widg_elems]
                for wk, wv in zip(widget_keys, chip_widg_elems):
                    dict_all_chip_widgets = {**dict_all_chip_widgets, **{wk : wv}}
                
            chip_blocks[filename] = {
                'widg_block': widg_block,
                'widget_keys': widget_keys,
                'widget_values' : widget_values,
                'chip_widgets': chip_widg_elems,
                'ui_params': temp_chip_ui.ui_params
            }
            chipblocks_list.append(widg_block)
        for widg_key, widget in dict_all_chip_widgets.items():
            widget.input(make_widget_change(widg_key), widget)
    except Exception as e:
        print("oh no.")
    return chip_blocks

def select_file(filename):
    global chip_settings, active_chip_path, active_chip_ui_path
    path = chip_path.format(name=filename)
    try:
        chip_settings = importlib.import_module(path)
        active_chip_path = path
        active_chip_ui_path = chip_ui_path.format(name=filename)
    except Exception as e:
        try:
            path = chip_path.format(name="default")
            chip_settings = importlib.import_module(path)
            active_chip_path = path
            active_chip_ui_path = chip_ui_path.format(name=filename)
        except Exception as e:
            print("You have no chips, choom")
        
    importlib.reload(chip_settings)
    return update_widget_visibility(filename)

def update_widget_visibility(filename, *blockslist):
    global chip_settings, widget_values, widget_keys, widgets, chipblocks_list, chip_blocks
    #chip_block_keys = chip_blocks.keys()
    visibility = []
    for chipname, chipinfo in chip_blocks.items():
        if filename == chipname:
            visibility.append(gr.update(visible=True))
            #blockitem.update(visible = True)  
            blockitem = chipinfo['widg_block']
            widget_keys = chipinfo['widget_keys']
            widget_values = chipinfo['widget_values']
            widgets = chipinfo['chip_widgets']
        else:
            visibility.append(gr.update(visible=False))
    return visibility                 
    
#A valiant, but ultimately failed attempt at modular interfaces
def populate_widgets(ui_params, mu, prefix=None):
    count = 0
    editable_elems = []
    ui_elems = []
    join = "" if prefix == None else "_-_"
    widget_values = []
    widget_keys = []
    sub_ui_widgets = []
    chip_sub_widgets = []
    for key, value in ui_params.items():
        widget_attributes = {}     
        
        container_type = getattr(gr, value['container_type']) if 'container_type' in value else None
        
        sub_ui_widgets = []
        chip_sub_widgets = []
            
        if 'attributes' in value:
            widget_attributes = value['attributes']
            widget_attributes['visible'] = True
            widget_attributes['interactive'] = not mu
            widgval = widget_attributes['start'] if 'start' in widget_attributes else widget_attributes['value']
            widget_values = [*widget_values, widgval]
        widget_type = getattr(gr,value['widget_type']) if 'widget_type' in value else None        
        
        widget_key= [f"{prefix}{join}{key}"] if widget_type is not None else []
        widget = [widget_type(**widget_attributes)] if widget_type is not None else []
        editable_elems = [*editable_elems, *widget]
        widget_keys = [*widget_keys, *widget_key]
        
        if 'sub_attr' in value:
            sub_attr = value['sub_attr']
            sub_key = f"{prefix}{join}{key}"
            if container_type is not None:
                with container_type() as container:             
                    chip_sub_widgets, sub_attr_keys, sub_attr_values, sub_ui_widgets = populate_widgets(sub_attr, mu, sub_key)
                sub_ui_widgets = [container]
            else:
                chip_sub_widgets, sub_attr_keys, sub_attr_values, sub_ui_widgets = populate_widgets(sub_attr, mu, sub_key)
                sub_ui_widgets = [*sub_ui_widgets]
            
            editable_elems = [*editable_elems, *chip_sub_widgets]
            widget_keys = [*widget_keys, *sub_attr_keys]
            widget_values = [*widget_values, *sub_attr_values]
            
        ui_elems = [*ui_elems, *widget, *sub_ui_widgets]            
        

    return editable_elems, widget_keys, widget_values, ui_elems

def traverse_to(from_obj, path):
    if len(path) == 1 and path[0] in from_obj:
        return from_obj[path[0]]
    
    newpath = path if path[0] in from_obj else ['sub_attr', *path]
    return traverse_to(from_obj[newpath[0]], newpath[1:])
    

def make_widget_change(key):
    def widget_change(widget_val):
        global dict_all_chip_widgets, chip_blocks
        traverse_path = key.split("_-_")
        ui_params = chip_blocks[traverse_path[0]]['ui_params']
        obj = traverse_to(ui_params, traverse_path[1:])
        obj['attributes']['value'] = widget_val
        # print(str(index) + ": " + str(slider))
        
    return widget_change

def on_switch_change(value):
    ui_settings['on'] = value
    
def output_prompts_change(value):
    ui_settings['output_prompts'] = value
    
def sample_other_prompts_change(value):
    ui_settings['sample_other_prompts'] = value
        
# I'm learning gradio with this function, bear with me here
def ui():
    global widgets, gradio, chip_settings, chipblocks_list
    mu = shared.args.multi_user    
    with gr.Row():
        with gr.Column():
            with gr.Row():
                gradio['on_switch'] = gr.Checkbox(label="Activate Brain-Hacking Chip", value=True)
            with gr.Row():
                gradio['output_prompts'] = gr.Checkbox(label="Debug: Output Prompts", value=False, info='Print all prompts to the console.')
                
                # This isn't working now and I'm not sure why, made it invisible for now
                gradio['sample_other_prompts'] = gr.Checkbox(label="Debug: Sample Other Prompts", value=False, info='Samples tokens from any extra prompts and prints their output to the console.', visible=False)            
        with gr.Column():
            with gr.Row():
                gradio['file_select'] = gr.Dropdown(choices=get_available_files(), value=None, label='Settings File', elem_classes='slim-dropdown', interactive=not mu)  
                create_refresh_button(gradio['file_select'], lambda: None, lambda: {'choices': get_available_files()}, 'refresh-button', interactive=not mu)
            with gr.Row():
                # Currently not using this, making it invisible for now... Not sure if I want script editing in here or not
                gradio['file_text'] = gr.Textbox(value='', lines=27, elem_id='textbox-notebook', elem_classes=['textbox', 'add_scrollbar'], visible=False)
    
    with gr.Row():
        widget_containers_full = make_chip_blocks(mu)
        
    gradio['file_select'].change(fn=select_file, inputs=gradio['file_select'], outputs=chipblocks_list)   
    # This doesn't trigger the change event... Why? It sets the value in the GUI, but I can't trigger the change event
    # gradio['file_select'].value = "default"
    
    gradio['on_switch'].change(on_switch_change, gradio['on_switch'])
    gradio['output_prompts'].change(output_prompts_change, gradio['output_prompts'])
    gradio['sample_other_prompts'].change(sample_other_prompts_change, gradio['sample_other_prompts'])
    
def custom_generate_chat_prompt(user_input, state, **kwargs):
    global ui_settings, chip_settings
    ui_params = None # EGjoni already has ui_params in chip_settings being updated, this is deprecated
    
    chip = importlib.import_module("extensions.BrainHackingChip.chip")
    importlib.reload(chip)
    
    if not chip_settings: # Just in case
        chip_settings = importlib.import_module("extensions.BrainHackingChip.chips.default.chip_settings")
        importlib.reload(chip_settings)
        
    prompt = chip.gen_full_prompt(chip_settings, ui_settings, ui_params, user_input, state, **kwargs)
    
    return prompt
