import gradio as gr
# These are parameters for the settings tab in the UI
# The user set values will be passed into the settings function, using the dictionary names from params
ui_params = {
    'weight' : {
        'attributes' : dict(label="Weight", value=0.3, minimum=0.0, maximum=1.0, step=0.05),
        'widget_type': "Slider"
    }
}
