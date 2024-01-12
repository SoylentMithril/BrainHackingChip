

class HackingchipSettings:
    def __init__(self, layer_count, attn_layers):
        self.layer_settings = [None] * layer_count
        self.attn_settings = [None] * len(attn_layers)
        
        self.attn_to_layers = attn_layers # Stores the layer index of each attention layer, for conversion from attention layer idx to layer idx
        self.layers_to_attn = [None] * layer_count # Stores the attention index of each layer, for conversion from layer idx to attention layer idx
        for index, value in enumerate(self.attn_to_layers): self.layers_to_attn[value] = index

class Value:
    def __init__(self, name=None, description=None, start=None, min=None, max=None, step=None):
        self.name = name
        self.description = description
        self.start = start
        self.min = min
        self.max = max
        self.step = step

class LayerSettings:
    def __init__(self, weight=0.0, cfg_func=None):
        self.weight = weight
        self.cfg_func = cfg_func
        
class VectorSettings:
    def __init__(self, weight=0.0, cfg_func=None):
        self.weight = weight
        self.cfg_func = cfg_func
        
class AttnSettings:
    def __init__(self, 
                 h=None,# VectorSettings provided with / requesting hidden state vectors inputting into attention layer
                 h_post= None, # VectorSettings provided with / requesting hidden state vectors after qkv calc (useful if, for example, you want to restore them to the way they were before attn.h was called)
                 q_in=None, # VectorSettings provided with / requesting incoming query vectors
                 q_in_post=None, # VectorSettings provided with / requesting query input vectors after attention output calc (useful if, for example, you want to restore them to the way they were before attn.q_in was called)
                 k_in=None, # VectorSettings provided with just incoming (not yet cached) key vectors. 
                 k_all=None, # VectorSettings provided with past(cached) + incoming (not yet cached) key vectors. 
                 k_all_post=None, # VectorSettings provided with / requesting key combined vectors after attention output calc (useful if, for example, you want to restore them to the way they were before attn.k_all was called)
                 v_in=None, # VectorSettings provided with incoming (not yet cached) value vectors.
                 v_all=None, # VectorSettings provided with past(cached) + incoming (not yet cached) value vectors.
                 v_all_post=None, # VectorSettings provided with / requesting value combined vectors after attention output calc (useful if, for example, you want to restore them to the way they were before attn.v_all was called)
                 a_ho=None, # VectorSettings provided with result of attention * value vectors of each head, prior to concatenation
                 a_c=None, # VectorSettings provided with concatenation of above, prior to attention output projection
                 a_po=None): # VectorSettings provided with attention output projection of above, prior to feed forward
        self.h = h 
        self.h_post = h_post
        self.q_in = q_in   
        self.q_in_post = q_in_post       
        self.k_in = k_in
        self.k_all = k_all
        self.k_all_post = k_all_post
        self.v_in = v_in
        self.v_all = v_all
        self.v_all_post = v_all_post
        self.a_ho = a_ho
        self.a_c = a_c
        self.a_po = a_po
        
        