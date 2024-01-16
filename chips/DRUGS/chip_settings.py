import torch
import gradio as gr
from extensions.BrainHackingChip.settings_classes import LayerSettings, AttnSettings, VectorSettings, Value
from .chip_ui import *
from .utils import get_perturbed_vectors, get_slice
from ...util.ui_helpers import traverse_to

sink_protect = 6
end_protect = {}
contingent_only = True


def get_tissue_bioavailability(idx, total_layers, dose_shaper, interpolation_type):
  layer_depth = idx/total_layers
  av1 = dose_shaper[0]
  av2 = dose_shaper[len(dose_shaper)-1]
  loop = True
  
  if layer_depth <= av1['depth']:
      av2 = av1
      loop = False
  if layer_depth >= av2['depth']:
      av1 = av2
      loop = False
  
  if loop:
      for d1, d2 in zip(dose_shaper, dose_shaper[1:]):
          if d1['depth'] <= layer_depth and d2['depth'] >= layer_depth:
              av1 = d1
              av2 = d2
              break
  
  equal = False
  if av1['depth'] == av2['depth']:
      equal = True       
  
  if interpolation_type == 'ceil' or equal:
      return max(av1['peakratio'], av2['peakratio'])
  elif interpolation_type == 'floor':
      return min(av1['peakratio'], av2['peakratio'])
  else:
      av1d = av1['depth']
      av2d = av2['depth']
      av1pr = av1['peakratio']
      av2pr = av2['peakratio']
      ldr = (layer_depth-av1d) / (av2d - av1d)
      return (ldr*(av2pr - av1pr)) + av1pr

def do_end_protect(key):
  return traverse_to(ui_params, [key,'type_globals','options'])

def get_shaped_dose(key, block_idx, total_blocks):
  if key not in active_noise_shapes: return 0
  base_theta = traverse_to(ui_params, [key,'type_globals','dose_theta'])
  shape_obj = active_noise_shapes[key]
  if len(shape_obj['shape]']) == 0: return 0
  interpolation_mode = shape_obj['mode']
  shape_arr = shape_obj['shape']
  shaped_dose = base_theta * get_tissue_bioavailability(block_idx, total_blocks, shape_arr, interpolation_mode)
  return shaped_dose

def brainhackingchip_settings(chip, params, last_kv_layer, head_layer):
  
  def dose_H(tensor, settings, hackingchip, block_idx=None, total_blocks= None, dim_info=None, total_layers=None, past_len = 0, cache=None,  **kwargs):
    start_idx  = sink_protect - past_len if past_len < sink_protect else 0
    end_protect[block_idx] = tensor.shape[dim_info['seq_vec']]     
    end_idx = tensor.shape[dim_info['seq_vec']] if not do_end_protect('h') else (tensor.shape[dim_info['seq_vec']]-end_protect[block_idx])+1  
    if end_idx > start_idx:
      sliced = get_slice(tensor, dim_info['seq_vec'], start_idx, end_idx)
      tensor[sliced] = get_perturbed_vectors(tensor[sliced], get_shaped_dose('h', block_idx, total_blocks))
    return tensor
    
  def dose_Q(tensor, settings, hackingchip, block_idx=None, total_blocks= None, dim_info=None, total_layers=None, past_len = 0, cache=None,  **kwargs):
    start_idx  = sink_protect - past_len if past_len < sink_protect else 0
    end_idx = tensor.shape[dim_info['seq_vec']] if not do_end_protect('q_in') else (tensor.shape[dim_info['seq_vec']]-end_protect[block_idx])+1  
    if end_idx > start_idx:
      sliced = get_slice(tensor, dim_info['seq_vec'], start_idx, end_idx)
      tensor[sliced] = get_perturbed_vectors(tensor[sliced], get_shaped_dose('q_in', block_idx, total_blocks))  
    return tensor

  def dose_K(tensor, settings, hackingchip, block_idx=None, total_blocks= None, dim_info=None, total_layers=None, past_len = 0, cache=None, **kwargs):
    start_idx  = sink_protect - past_len if past_len < sink_protect else sink_protect
    end_idx = tensor.shape[dim_info['seq_vec']] if not do_end_protect('k_all') else tensor.shape[dim_info['seq_vec']]-end_protect[block_idx]      
    if end_idx > start_idx:
      sliced = get_slice(tensor, dim_info['seq_vec'], start_idx, end_idx)
      tensor[sliced] = get_perturbed_vectors(tensor[sliced], get_shaped_dose('k_all', block_idx, total_blocks))
    return tensor

  def dose_V(tensor, settings, hackingchip, block_idx=None, total_blocks= None, dim_info=None, total_layers=None, past_len = 0, **kwargs):
    start_idx  = sink_protect - past_len if past_len < sink_protect else sink_protect
    end_idx = tensor.shape[dim_info['seq_vec']] if not do_end_protect('v_all') else tensor.shape[dim_info['seq_vec']]-end_protect[block_idx]      
    if end_idx > start_idx:
      sliced = get_slice(tensor, dim_info['seq_vec'], start_idx, end_idx)
      tensor[sliced] = get_perturbed_vectors(tensor[sliced], get_shaped_dose('v_all', block_idx, total_blocks))
    return tensor
  
  def dose_A(tensor, settings, hackingchip, block_idx=None, total_blocks= None, dim_info=None, total_layers=None, past_len = 0, **kwargs):
    start_idx  = sink_protect - past_len if past_len < sink_protect else 0
    end_idx = tensor.shape[dim_info['seq_vec']] if not do_end_protect('a_ho') else (tensor.shape[dim_info['seq_vec']]-end_protect[block_idx])+1
    if end_idx > start_idx:
      sliced = get_slice(tensor, dim_info['seq_vec'], start_idx, end_idx)
      tensor[sliced] = get_perturbed_vectors(tensor[sliced], get_shaped_dose('a_ho', block_idx, total_blocks))
    return tensor
  
  attn_test = AttnSettings()

  # Prepare H, Q, K, V, A function hooks
  
  attn_test.h = VectorSettings(cfg_func=dose_H)
  attn_test.q_in = VectorSettings(cfg_func=dose_Q)
  #attn_test.k_in = hm
  attn_test.k_all = VectorSettings(cfg_func=dose_K)
  #attn_test.v_in = hmm
  attn_test.v_all = VectorSettings(cfg_func=dose_V)
  attn_test.a_ho = VectorSettings(cfg_func=dose_A)
  #attn_test.a_c = hmmm
  #attn_test.a_po = HMMMMM
  chip.attn_settings = [attn_test] * len(chip.attn_settings) # testing every attention layer

  return chip