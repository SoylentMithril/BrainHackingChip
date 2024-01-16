import gradio as gr 
from modules.html_generator import get_image_cache
from pathlib import Path
import copy


custom_css = """
.input_noise_chkbx label::before, 
.early_noise_chkbx label::before,
.mid_noise_chkbx label::before,
.late_noise_chkbx label::before,
.output_noise_chkbx label::before {
  content: "";
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-size: 100% 100%;
  opacity: 0.5;
  z-index: 1;
  background-color: #0005;
  filter: invert(0);
}

.input_noise_chkbx label::before {
background-image: url('file/""" + get_image_cache(Path('extensions/BrainHackingChip/chips/DRUGS/css/input_noise.gif')) + """');

}

.early_noise_chkbx label::before {
background-image: url('file/""" + get_image_cache(Path('extensions/BrainHackingChip/chips/DRUGS/css/early_noise.gif')) + """');
}

.mid_noise_chkbx label::before {
background-image: url('file/""" + get_image_cache(Path('extensions/BrainHackingChip/chips/DRUGS/css/mid_noise.gif')) + """');
}

.late_noise_chkbx label::before {
background-image: url('file/""" + get_image_cache(Path('extensions/BrainHackingChip/chips/DRUGS/css/late_noise.gif')) + """');
}

.output_noise_chkbx label::before {
background-image: url('file/""" + get_image_cache(Path('extensions/BrainHackingChip/chips/DRUGS/css/output_noise.gif')) + """');
}
.noise_bx label {
  height: 75px;
  background-size: 100% 100%;
  width: 100%;
}
.noise_bx label:has(> input[type='checkbox']:checked) {
  filter: invert(1);
}

"""

input_noise_profile = {
  'mode' : 'ceil',
  'shape': [
    {'depth': 0.01,'peakratio' : 1.0},
    {'depth': 0.05,'peakratio' : 0.0, 'skip_after': True} #for easily combining noise shape segments
  ],
}

early_noise_profile = {
  'mode' : 'interpolate',
  'shape': [
    {'depth': 0.00,'peakratio' : 0.0, 'skip_before' : True},
    {'depth': 0.05,'peakratio' : 1.0},
    {'depth': 0.30,'peakratio' : 1.0},
    {'depth': 0.35,'peakratio' : 0.0, 'skip_after' : True}
  ],
}

mid_noise_profile = {
  'mode' : 'interpolate',
  'shape': [
    {'depth': 0.30,'peakratio' : 0.0, 'skip_before' : True},
    {'depth': 0.35,'peakratio' : 1.0},
    {'depth': 0.60,'peakratio' : 1.0},
    {'depth': 0.65,'peakratio' : 0.0, 'skip_after' : True}
  ],
}

late_noise_profile = {
  'mode' : 'interpolate',
  'shape': [
    {'depth': 0.60,'peakratio' : 0.0, 'skip_before' : True},
    {'depth': 0.65,'peakratio' : 1.0},
    {'depth': 0.95,'peakratio' : 1.0},
    {'depth': 1.00,'peakratio' : 0.0, 'skip_after' : True}
  ],
}

output_noise_profile = {
  'mode' : 'ceil',
  'shape': [
    {'depth': 0.99,'peakratio' : 0.0,'skip_before': True},
    {'depth': 1.0,'peakratio' : 1.0}
  ],
}


active_noise_shapes = {}

def construct_noise_shape(selected_noise_types):
  mode = 'ceil'
  frame_arr = []
  total_entries = len(selected_noise_types)
  last_sib_idx = -1
  for sib_idx, n_type in selected_noise_types:
    mode = 'interpolate' if 'mode' in n_type and n_type['mode'] == 'interpolate' else mode
  
  for idx, (sib_idx, n_type) in enumerate(selected_noise_types):
    for noise_frame in n_type['shape']:
      if 'skip_before' in noise_frame and last_sib_idx == sib_idx-1:
        continue
      if 'skip_after' in noise_frame and idx < total_entries-1 and sib_idx + 1 == selected_noise_types[idx+1]:
        continue      
      frame_arr.append(noise_frame)
      last_sib_idx = sib_idx
  return {'mode': mode, 'shape': frame_arr}


def update_active_noise_shapes(widget_ref = None, widget_val=None, traverse_path=None):
  if 'h' in traverse_path:
    noise_type = 'h'
  if 'q_in' in traverse_path:
    noise_type = 'q_in'
  if 'k_all' in traverse_path:
    noise_type = 'k_all'
  if 'v_all' in traverse_path:
    noise_type = 'v_all'
  if 'a_ho' in traverse_path:
    noise_type = 'a_ho'
  
  noise_siblings = ui_params[noise_type]['sub_attr']['shape_sliders']['sub_attr']
  selected_noise_items = []
  for idx, (shape_name, attr) in enumerate(noise_siblings.items()):
    drug_profile = attr['drug_profile']
    if attr['attributes']['value'] is True:
      selected_noise_items.append((idx, drug_profile))
  
  active_noise_shapes[noise_type] = construct_noise_shape(selected_noise_items)
    
def make_noise_type(attr_dict, noise_profile):
  return {
    'container_type': "Column",
    'attributes': attr_dict,
    'widget_type': 'Checkbox',
    'callback' : 'update_active_noise_shapes',
    "drug_profile": noise_profile
  }

def make_drug_type(attr_dict):
  return {
    'container_type': "Column",
    'sub_attr': {
      'type_globals': {
          'container_type': "Row",
          "sub_attr": {
            'dose_theta' : {
              'container_type': "Column",
                'attributes': attr_dict,
                'widget_type': "Slider",
              },
            'options' : {
              'container_type': "Column",
              'attributes': dict(label="Contingent only", value=True, info="If checked, model will have fully lucid understanding of the last thing user said."),
              'widget_type': "Checkbox",                    
            }  
          }
        },
       'shape_sliders': { 
          'container_type': "Row",
          'sub_attr': {
            'input_noise' : make_noise_type(dict(label='noise input layers', value=False, 
                              elem_classes=["input_noise_chkbx", "noise_bx"]), input_noise_profile),
            'early_noise' : make_noise_type(dict(label='noise early layers', 
                              value=False, elem_classes=["early_noise_chkbx", "noise_bx"]), early_noise_profile),
            'mid_noise' : make_noise_type(dict(label='noise mid layers', 
                              value=False, elem_classes=["mid_noise_chkbx", "noise_bx"]),mid_noise_profile),
            'late_noise' : make_noise_type(dict(label='noise late layers', 
                              value=False, elem_classes=["late_noise_chkbx", "noise_bx"]),late_noise_profile),
            'output_noise' : make_noise_type(dict(label='noise output layers', 
                              value=False, elem_classes=["output_noise_chkbx", "noise_bx"]), output_noise_profile)
          }
        }
      }
    }

ui_params = {
  'a_ho': make_drug_type(dict(label="A dose", value=0.1, minimum=0.0, maximum=1.0, step=0.01)),
  'v_all': make_drug_type(dict(label="V dose", value=0.1, minimum=0.0, maximum=1.0, step=0.01)),
  'k_all': make_drug_type(dict(label="K dose", value=0.1, minimum=0.0, maximum=1.0, step=0.01)),
  'q_in': make_drug_type(dict(label="Q dose", value=0.1, minimum=0.0, maximum=1.0, step=0.01)),
  'h': make_drug_type(dict(label="H dose", value=0.1, minimum=0.0, maximum=1.0, step=0.01))
}

