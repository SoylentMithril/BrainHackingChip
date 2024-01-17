import importlib
import gradio as gr
from modules import shared, chat, ui, ui_chat, ui_default, ui_file_saving, ui_model_menu, ui_notebook, ui_parameters, ui_session, models_settings, training
import server

from modules.ui import create_refresh_button, list_interface_input_elements
from .util.ui_helpers import traverse_to
import os
import fnmatch

import inspect


import json

params = {
    'is_tab': True,
    'display_name': 'Brain-Hacking Chip'
}

gradio = {}
selected_files = []
default_max_widgets = 10 # This will probably have to be higher
default_max_widgets = 10
chip_path = "extensions.BrainHackingChip.chips.{name}.chip_settings"
chip_ui_path = "extensions.BrainHackingChip.chips.{name}.chip_ui"
active_chip_path = "extensions.BrainHackingChip.chips.default.chip_settings" # not using this with multichip
active_chip_ui_path = "extensions.BrainHackingChip.chips.default.chip_ui" # not using this with multichip
chip_blocks = {}
chipblocks_list = []
# Globals
chip_settings = []
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
            css_str = temp_chip_ui.custom_css if hasattr(temp_chip_ui, 'custom_css') else None            
            with gr.Group(visible=False) as widg_block:
                if css_str is not None:
                    gr.HTML(f"""<style>{css_str}</style>""")
                chip_widg_elems, widget_keys, widget_values, ui_components = populate_widgets(temp_chip_ui.ui_params, mu, filename)
                all_chip_widgets = [*all_chip_widgets, *chip_widg_elems]
                for wk, wv in zip(widget_keys, chip_widg_elems):
                    dict_all_chip_widgets = {**dict_all_chip_widgets, **{wk : wv}}
            
            chip_blocks[filename] = {
                'widg_block': widg_block,
                'widget_keys': widget_keys,
                'widget_values' : widget_values,
                'chip_widgets': chip_widg_elems,
                'chip_ui' : temp_chip_ui
            }            
            chipblocks_list.append(widg_block)
        for widg_key, widge_ref in dict_all_chip_widgets.items():
            widge_ref.change(make_widget_change(widg_key, widge_ref), widge_ref)
    except Exception as e:
        print("oh no.")
        
    update_shared_gradio_refs(dict_all_chip_widgets)
    
    return chip_blocks

def select_file(filenames):
    global chip_settings, active_chip_path, active_chip_ui_path, selected_files
    selected_files = filenames
    chip_settings = [] # eventually could avoid reloading already loaded things
    try:
        for filename in filenames:
            path = chip_path.format(name=filename)
            chip_settings.append(importlib.import_module(path))
            # active_chip_path = path
            # active_chip_ui_path = chip_ui_path.format(name=filename)
    except Exception as e:
        try:
            if not chip_settings:
                path = chip_path.format(name="default")
                chip_settings = importlib.import_module(path)
                # active_chip_path = path
                # active_chip_ui_path = chip_ui_path.format(name=filename)
        except Exception as e:
            print("You have no chips, choom")
        
    for chip_settings_single in chip_settings: importlib.reload(chip_settings_single)
    return update_widget_visibility(filenames)

def update_widget_visibility(filenames, *blockslist):
    global widget_values, widget_keys, widgets, chipblocks_list, chip_blocks
    #chip_block_keys = chip_blocks.keys()
    visibility = []
    for chipname, chipinfo in chip_blocks.items():
        if chipname in filenames:
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

def make_widget_change(key, widge_ref):
    def widget_change(widget_val):
        global dict_all_chip_widgets, chip_blocks
        traverse_path = key.split("_-_")
        chip_ui = chip_blocks[traverse_path[0]]['chip_ui']
        if chip_ui.ui_params is not None:
            ui_params = chip_ui.ui_params
            obj = traverse_to(ui_params, traverse_path[1:], False)
            obj['attributes']['value'] = widget_val
            if 'callback' in obj:
                if callable(obj['callback']): callback = obj['callback']
                else: callback = getattr(chip_ui, obj['callback'])
                callback(widget_ref = widge_ref, widget_val=widget_val, traverse_path=traverse_path)
        # print(str(index) + ": " + str(slider))        
    return widget_change

def on_switch_change(value):
    ui_settings['on'] = value
    
def output_prompts_change(value):
    ui_settings['output_prompts'] = value
    
def sample_other_prompts_change(value):
    ui_settings['sample_other_prompts'] = value
    
# Using this to pass params into chips in a less verbose way, and can also be used for saving
def get_widget_params(widgets):
    params = {}
    
    for key, widget in widgets.items():
        info = {}
        
        if 'attributes' in widget:
            if 'value' in widget['attributes']: # will there ever be other things to save?
                info['value'] = widget['attributes']['value']
                # print(key + ": " + str(widget['attributes']['value'])) # save widget['attributes']['value']
            
        if 'sub_attr' in widget:
            info['sub'] = get_widget_params(widget['sub_attr'])
            
        if len(info) > 0:
            params[key] = info
            
    return params

def get_chip_params():
    global chip_blocks, gradio, selected_files
    
    params = []
    
    for selected_file in selected_files:
        if selected_file and selected_file in chip_blocks and 'chip_ui' in chip_blocks[selected_file]:
            params.append(get_widget_params(chip_blocks[selected_file]['chip_ui'].ui_params))
        else: # This shouldn't ever happen, but things get really messed up if ui_params len isn't chip_blocks len
            params.append({})
    
    return params

settings_path = './extensions/BrainHackingChip/{filename}.json'

def save_settings_click():
    global settings_path, chip_blocks
    settings = {}
    
    for key, value in chip_blocks.items():
        settings[key] = get_widget_params(value['ui_params'])
        
    try:
        with open(os.path.join(os.getcwd(), settings_path.format(filename="default_settings")), 'w') as json_file:
            json.dump(settings, json_file) 
    except Exception as e:
        print("Failed to save settings")
                   
# From what I know, this should work, not sure what I'm missing
def load_settings_click(chip_filename):
    # Currently doing global settings, but could do settings for each chip in their own folders too
    global settings_path, widget_keys, widgets
    
    settings = {}
    
    join = "_-_"
    
    try:
        with open(os.path.join(os.getcwd(), settings_path.format(filename="default_settings")), 'r') as json_file:
            settings = json.load(json_file)    
    except Exception as e:
        print("Failed to load settings")
        
    settings_by_tag = {}
        
    def build_settings_strings(params, prefix):
        for param, info in params.items():
            next_prefix = prefix + join + param
            
            if 'value' in info:
                settings_by_tag[next_prefix] = info['value']
            
            if 'sub' in info:
                build_settings_strings(info['sub'], next_prefix)
                
    for chip_name, chip_settings in settings.items():
        build_settings_strings(chip_settings, chip_name)
        
    settings_out = []
        
    for key in widget_keys:
        if key in settings_by_tag:
            settings_out.append(gr.update(value=settings_by_tag[key]))
        else:
            settings_out.append(gr.update())
        
    return settings_out

all_shared_widget_keys = []


# Once state is generated with all the BHC values in it, I need to extract them for easy use
def get_state_ui_params(state):
    # Ok, so I return a list, with an entry for each file in order, so start with a dict with a filename for each for now
    # Then the value is a dict of names
    # Each name is a dict that may have a value or sub, and sub will contain a dict of names
    
    # use these to order it into an array? 
    filenames_in_order = get_available_files()
    
    bhc = "bhc"
    join = "_-_"
    bhc_prefix = bhc + join
    
    params_root = {}
    
    # Example: params_root['DRUGS']['sub']
    
    def parse_key(key, value, params):
        parts = key.split(join, 1)
        
        if len(parts) == 2: # can't place value yet
            this_parent = parts[0]
            next_key = parts[1]
            
            if not this_parent in params:
                params[this_parent] = {}
                
            this_params = params[this_parent]
            
            if not 'sub' in this_params:
                this_params['sub'] = {}
            
            parse_key(next_key, value, this_params['sub'])
        else: # can place value
            if not key in params:
                params[key] = {}
                
            params[key]['value'] = value
    
    for key, value in state.items():
        if key.startswith(bhc_prefix):
            next_key = key[len(bhc_prefix):]
            parse_key(next_key, value, params_root)
     
    # Now to put it all in order in the list
    params_list = [params_root[filename]['sub'] if filename in params_root else {} for filename in filenames_in_order]
            
    return params_list
    
    
# After generating widgets in here, I call this and put them all in shared.gradio and do other supporting things
def update_shared_gradio_refs(all_widgets):
    global all_shared_widget_keys
    
    # doing all widgets so I don't have to change this much
    
    bhc_prefix = "bhc_-_"
    bhc_format = bhc_prefix + "{key}"
    
    # first have to remove all bhc references in shared.gradio
    shared.gradio = {key: value for key, value in shared.gradio.items() if not key.startswith(bhc_prefix)}
    
    # this was another way I was removing old references, but the above is probably better
    # for widget_key in all_shared_widget_keys:
    #     if widget_key in shared.gradio:
    #         del shared.gradio[widget_key]
    
    # this may have already been set, just going to set it again
    # it shouldn't be set with the current code, but I'll just leave this anyway
    all_shared_widget_keys = []
    
    # Putting every widget in shared.gradio and storing the key used for it in all_shared_widget_keys
    for key, value in all_widgets.items():
        bhc_key = bhc_format.format(key=key)
        all_shared_widget_keys.append(bhc_key)
        shared.gradio[bhc_key] = value
                
    # Need to update this or bad things happen
    shared.input_elements = hijack_list_interface_input_elements()
    
    # TODO: If the below func() calls get run more than once, I'm sure it will be very bad
    # They add all the UI event handlers in ooba among other things, so this function really shouldn't be called twice
    
    # TODO: Ok, for the below, I need to separate the function lists
    # The create_ui thing needs to run during extension tab creation or something
    # Not quite sure how to fix that
    # Right now, session tab ends up in this BHC tab
    for func in original_create_ui:
        None # Not running func() currently because it will be placed in the BHC tab
        # func()
        
    # Ok, and now I need to call the rest of the UI setup functions, it will be useless without these
    for func in original_event_handlers:
        func()
        
original_func = list_interface_input_elements
           
def hijack_list_interface_input_elements():
    global all_shared_widget_keys
    
    elements = original_func()
    
    # putting all bhc shared.gradio keys in, this will make ooba process the widget values
    elements += all_shared_widget_keys
    
    
    return elements
    
def hijack_and_do_nothing():
    # print("test")
    None

def setup():
    global all_shared_widget_keys, chip_ui_path, original_event_handlers, original_create_ui
    
    # It's time to hijack some functions in the normal webui code so we can hook into the state gathering events
    # However, this makes other parts of the webui code angry, so we have to hijack them too and make them chill awhile
    
    # Using inspect to target every usage of ui
    
    all_modules = [m for m in globals().values() if inspect.ismodule(m)]
    
    hijack_ui_map = {
        'list_interface_input_elements': hijack_list_interface_input_elements,
    }
    
    for module in all_modules:
        if hasattr(module, 'ui'):
            ui_module = getattr(module, 'ui')
            # print(module.__name__) # debugging for now
            for func_name, hijack_func in hijack_ui_map.items():
                if hasattr(ui_module, func_name) and callable(getattr(ui_module, func_name)):
                    setattr(ui_module, func_name, hijack_func) 
                   
    # Ok, and then all of these functions will break if they run before this extension's ui() function runs
    # So it's time for them to take a little nap until that happens       
 
    delay_create_ui = [
        ui_session
    ]

    delay_create_event_handlers = [
        ui_chat, 
        ui_default,
        ui_notebook, 
        ui_file_saving,
        ui_parameters,
        ui_model_menu
        ]
    
    # This is what will run them again
    original_event_handlers = [delay.create_event_handlers for delay in delay_create_event_handlers]
    
    # TODO: I'm not actually calling this anywhere because it puts the created UI inside the BHC tab! I'll keep thinking of a fix
    original_create_ui = [delay.create_ui for delay in delay_create_ui]
    
    for delay in delay_create_ui:
        delay.create_ui = hijack_and_do_nothing 
        
    for delay in delay_create_event_handlers:
        delay.create_event_handlers = hijack_and_do_nothing
    
            
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
            with gr.Row():
                # Hiding save/load while figuring out loading
                gradio['save_settings_button'] = gr.Button("Save Settings", visible=False)
                gradio['load_settings_button'] = gr.Button("Load Settings", visible=False)
        with gr.Column():
            with gr.Row():
                gradio['file_select'] = gr.Dropdown(choices=get_available_files(), value=["default"], label='Settings File', elem_classes='slim-dropdown', multiselect=True, interactive=not mu)  
                create_refresh_button(gradio['file_select'], lambda: None, lambda: {'choices': get_available_files()}, 'refresh-button', interactive=not mu)
            with gr.Row():
                # Currently not using this, making it invisible for now... Not sure if I want script editing in here or not
                gradio['file_text'] = gr.Textbox(value='', lines=27, elem_id='textbox-notebook', elem_classes=['textbox', 'add_scrollbar'], visible=False)
    
    with gr.Row():
        widget_containers_full = make_chip_blocks(mu)
        
    gradio['file_select'].change(fn=select_file, inputs=gradio['file_select'], outputs=chipblocks_list)   
    
    gradio['on_switch'].change(on_switch_change, gradio['on_switch'])
    gradio['output_prompts'].change(output_prompts_change, gradio['output_prompts'])
    gradio['sample_other_prompts'].change(sample_other_prompts_change, gradio['sample_other_prompts'])
    
    gradio['save_settings_button'].click(save_settings_click)
    
    # Not sure how to load yet... Have a function that outputs gr.updates for widgets, but the update doesn't happen
    gradio['load_settings_button'].click(fn=load_settings_click, inputs=gradio['file_select'], outputs=widgets)
    
    # Auto-load GUI widgets, will change this to load settings file once that's setup
    # This solution isn't perfect though, it requires the user to visit the UI first
    # So if a user has the UI up, closes the backend and restarts it, then interacts with the UI without reloading, the below event will not have triggered
    shared.gradio['interface'].load(fn=select_file, inputs=gradio['file_select'], outputs=chipblocks_list)
    
def custom_generate_chat_prompt(user_input, state, **kwargs):
    global ui_settings, chip_settings
    
    ui_params = get_state_ui_params(state)
    
    chip = importlib.import_module("extensions.BrainHackingChip.chip")
    importlib.reload(chip)
    
    if not chip_settings: # Just in case
        chip_settings_default = importlib.import_module("extensions.BrainHackingChip.chips.default.chip_settings")
        importlib.reload(chip_settings_default)
        chip_settings.append(chip_settings_default)
        
    prompt = chip.gen_full_prompt(chip_settings, ui_settings, ui_params, user_input, state, **kwargs)
    
    return prompt
