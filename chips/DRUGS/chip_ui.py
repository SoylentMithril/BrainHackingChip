import gradio as gr 
import copy



just_sliders = {
    'shape_sliders': { 
        'container_type': "Row",
        'sub_attr': {
            '19' : {
                'container_type': "Column",
                'attributes': dict(label='depth 19%', value=0.0, minimum=0.0, maximum=1.0, step=0.01),
                'widget_type': "Slider"
            }, 
            '20' : {
                'container_type': "Column",
                'attributes': dict(label='depth 20%', value=1.0, minimum=0.0, maximum=1.0, step=0.01),
                'widget_type': "Slider"
            },
            '70' : {
                'container_type': "Column",
                'attributes': dict(label='depth 70%', value=1.0, minimum=0.0, maximum=1.0, step=0.01),
                'widget_type': "Slider"
            },
            '90' : {
                'container_type': "Column",
                'attributes': dict(label='depth 90%', value=0.0, minimum=0.0, maximum=1.0, step=0.01),
                'widget_type': "Slider"
            }
        }
    }
}
protect_and_sliders = {
   'options' : {
        'container_type': "Column",
        'attributes': dict(label="Contingent only", value=True, info="If checked, model will have fully lucid understanding of the last thing user said."),
        'widget_type': "Checkbox",
        
    }, **just_sliders
}

ui_params = {
  'h': {'container_type': "Column",
        'attributes': dict(label="H", value=0.05, minimum=0.0, maximum=1.0, step=0.01),
        'widget_type': "Slider",
        'sub_attr': copy.deepcopy(just_sliders)
    },
  'q_in': {
      'container_type': "Column",
      'attributes': dict(label="Q", value=0.1, minimum=0.0, maximum=1.0, step=0.01),
      'widget_type': "Slider",
      'sub_attr': copy.deepcopy(just_sliders)
    },
  'k_all': {'container_type': "Column",
        'attributes': dict(label="K", value=0.1, minimum=0.0, maximum=1.0, step=0.01),
        'widget_type': "Slider",
        'sub_attr': copy.deepcopy(protect_and_sliders)
    },
  'v_all': {
      'container_type': "Column",
      'attributes': dict(label="V", value=0.2, minimum=0.0, maximum=1.0, step=0.01),
      'widget_type': "Slider",
      'sub_attr': copy.deepcopy(protect_and_sliders)
    },
  'a_ho': {
      'container_type': "Column",
      'attributes': dict(label="A", value=0.3, minimum=0.0, maximum=1.0, step=0.01),
      'widget_type': "Slider",
      'sub_attr': copy.deepcopy(just_sliders)
    }
}

placeholder = """
  {'shape':[
    {'depth': 0.199, 'peak': 0.0},
    {'depth': 0.2, 'peak': 1.0},
    {'depth': 0.7, 'peak': 1.0},
    {'depth': 0.9, 'peak': 0.0}
    ],
    'mode': 'interpolate' /*or 'floor', or 'ceil'*/
  }
  """
"""default_shape_args = {
  'placeholder': placeholder,
  'visible' : True,
  'value': {'shape': [
    {'depth': 0.199, 'peak': 0.0},
    {'depth': 0.2, 'peak': 1.0},
    {'depth': 0.7, 'peak': 1.0},
    {'depth': 0.9, 'peak': 0.0}
    ],
    'mode': 'interpolate'
  }
}"""
