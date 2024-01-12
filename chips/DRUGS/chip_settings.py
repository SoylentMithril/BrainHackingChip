import torch
import gradio as gr
from extensions.BrainHackingChip.settings_classes import LayerSettings, AttnSettings, VectorSettings, Value
from .chip_ui import *
from .utils import get_perturbed_vectors, get_slice

sink_protect = 6
end_protect = {}
contingent_only = True


def do_end_protect(key):
  return ui_params[key]['sub_attr']['options']['attributes']['value']

def get_shaped_dose(key, layer_idx):
  #TODO: apply actual dose shaping
  return ui_params[key]['attributes']['value']

def brainhackingchip_settings(chip, params, last_kv_layer, head_layer):
  
    def dose_H(tensor, settings, hackingchip, layer_idx=None, dim_info=None, total_layers=None, cache=None,  **kwargs):
      seq_len = tensor.shape[dim_info['seq_vec']]
      sink = sink_protect if seq_len > 1 else 0
      if seq_len > sink:
        sliced = get_slice(tensor, dim_info['seq_vec'], sink, seq_len)
        tensor[sliced] = get_perturbed_vectors(tensor[sliced], get_shaped_dose('h', layer_idx))
      return tensor
     
    def Q_dose(tensor, settings, hackingchip, layer_idx=None, dim_info=None, total_layers=None, cache=None,  **kwargs):
      seq_len = tensor.shape[dim_info['seq_vec']]
      sink = sink_protect if seq_len > 1 else 0
      end_protect[layer_idx] = seq_len-sink
      if seq_len > sink:
        sliced = get_slice(tensor, dim_info['seq_vec'], sink, seq_len)
        tensor[sliced] = get_perturbed_vectors(tensor[sliced], get_shaped_dose('q_in', layer_idx))  
      return tensor

    def dose_K(tensor, settings, hackingchip, layer_idx=None, dim_info=None, total_layers=None, cache=None, **kwargs):
      end_idx = tensor.shape[dim_info['seq_vec']] if not do_end_protect('k_all') else tensor.shape[dim_info['seq_vec']]-end_protect[layer_idx]      
      if end_idx > sink_protect:
        sliced = get_slice(tensor, dim_info['seq_vec'], sink_protect, end_idx)
        tensor[sliced] = get_perturbed_vectors(tensor[sliced], get_shaped_dose('k_all', layer_idx))
      return tensor

    def dose_V(tensor, settings, hackingchip, layer_idx=None, dim_info=None, total_layers=None, **kwargs):
      end_idx = tensor.shape[dim_info['seq_vec']] if not do_end_protect('v_all') else tensor.shape[dim_info['seq_vec']]-end_protect[layer_idx]      
      if end_idx > sink_protect:
        sliced = get_slice(tensor, dim_info['seq_vec'], sink_protect, end_idx)
        tensor[sliced] = get_perturbed_vectors(tensor[sliced], get_shaped_dose('v_all', layer_idx))
      return tensor
    
    def dose_A(tensor, settings, hackingchip, layer_idx=None, dim_info=None, total_layers=None, **kwargs):
      end_idx = tensor.shape[dim_info['seq_vec']]
      sliced = get_slice(tensor, dim_info['seq_vec'], sink_protect, end_idx)
      if end_idx > sink_protect:
        tensor[sliced] = get_perturbed_vectors(tensor[sliced], get_shaped_dose('a_ho', layer_idx))
      return tensor
    
    attn_test = AttnSettings()

    # Prepare H, Q, K, V, A function hooks
    
    attn_test.h = VectorSettings(cfg_func=dose_H)
    attn_test.q_in = VectorSettings(cfg_func=Q_dose)
    #attn_test.k_in = hm
    attn_test.k_all = VectorSettings(cfg_func=dose_K)
    #attn_test.v_in = hmm
    attn_test.v_all = VectorSettings(cfg_func=dose_V)
    attn_test.a_ho = VectorSettings(cfg_func=dose_A)
    #attn_test.a_c = hmmm
    #attn_test.a_po = HMMMMM
    chip.attn_settings = [attn_test] * len(chip.attn_settings) # testing every attention layer

    return chip