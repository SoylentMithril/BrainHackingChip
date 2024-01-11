import torch
from extensions.BrainHackingChip.settings_classes import LayerSettings, AttnSettings, VectorSettings, Value

# These are parameters for the settings tab in the UI
# The user set values will be passed into the settings function, using the dictionary names from params
# Currently limited to a max of 10 params, but that number can be increased
ui_params = {
  'h': Value(name="H", start=0.2, min=0.0, max=1.0, step=0.05),
  'q': Value(name="Q", start=0.2, min=0.0, max=1.0, step=0.05),
  'k': Value(name="K", start=0.2, min=0.0, max=1.0, step=0.05),
  'v': Value(name="V", start=0.2, min=0.0, max=1.0, step=0.05),
  'a': Value(name="A", start=0.2, min=0.0, max=1.0, step=0.05),
}

def brainhackingchip_settings(chip, params, last_kv_layer, head_layer):
    def build_function(key):
        weight = params[key] if key in params else 0.0

        def drugs_func(tensor, settings, hackingchip):
                # This is where the actual drugs code would go! H, Q, K, V, A vectors will be passed in during inference according to attn_test

                # Modify vectors here stored in tensor and then return the modified values
                # Can use the user determined weight from the UI for this particular vector

                return tensor

        return drugs_func

    attn_test = AttnSettings()

    # Prepare H, Q, K, V, A function hooks

    attn_test.h = VectorSettings(cfg_func=build_function('h'))
    attn_test.q = VectorSettings(cfg_func=build_function('q'))
    attn_test.k = VectorSettings(cfg_func=build_function('k'))
    attn_test.v = VectorSettings(cfg_func=build_function('v'))
    attn_test.a = VectorSettings(cfg_func=build_function('a'))

    # Uncomment one line below, first line uses every single attention layer and 2nd line only uses the very last attention layer

    chip.attn_settings = [attn_test] * len(chip.attn_settings) # testing every attention layer
    # chip.attn_settings[-1] = attn_test # testing only the last attention

    return chip