

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
                 h=None,# VectorSettings provided with hidden state vectors inputting into attention layer
                 q_in=None,# VectorSettings provided with incoming query vectors
                 q_inf=None,# Flash_attention version of the above. Just a transpose on dim 1,2 of the above. It is safe to provide
                 #functions for both of these, only the relevant one will be called.
                 k_in=None, # VectorSettings provided with just incoming (not yet cached) key vectors. 
                 k_all=None, # VectorSettings provided with past(cached) + incoming (not yet cached) key vectors. 
                 k_allf=None, # VectorSettings provided with past(cached) + incoming (not yet cached) key vectors. 
                 v_in=None, # VectorSettings provided with incoming (not yet cached) value vectors.
                 v_all=None, # VectorSettings provided with past(cached) + incoming (not yet cached) value vectors.
                 v_allf=None,  # VectorSettings provided with past(cached) + incoming (not yet cached) value vectors.
                 a_ho=None, # VectorSettings provided with result of attention * value vectors of each head, prior to concatenation
                 a_c=None, # VectorSettings provided with concatenation of above, prior to attention output projection
                 a_po=None): # VectorSettings provided with attention output projection of above, prior to feed forward
        self.h = h 
        self.q_in = q_in 
        self.q_inf = q_inf
        self.k_in = k_in
        self.k_all = k_all
        self.k_allf = k_allf
        self.v_in = v_in
        self.v_all = v_all
        self.v_allf = v_allf
        self.a_ho = a_ho
        self.a_c = a_c
        self.a_po = a_po
        
        