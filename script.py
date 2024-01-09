import importlib
import gradio as gr

params = {
    'is_tab': False,
    'display_name': 'Brain-Hacking Chip'
}
            
def custom_generate_chat_prompt(user_input, state, **kwargs):
    chip = importlib.import_module("extensions.BrainHackingChip.chip")
    importlib.reload(chip)
    
    chip_settings = importlib.import_module("extensions.BrainHackingChip.chip_settings")
    importlib.reload(chip_settings)
    
    prompt = chip.gen_full_prompt(chip_settings, user_input, state, **kwargs)
    
    return prompt
