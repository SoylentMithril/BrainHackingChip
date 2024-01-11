

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
    def __init__(self, h=None, q=None, k=None, v=None, a=None): # Use VectorSettings
        self.h = h # VectorSettings for hidden state vectors inputting into attention layer
        self.q = q # VectorSettings for query vectors
        self.k = k # VectorSettings for key vector
        self.v = v # VectorSettings for value vectors
        self.a = a # VectorSettings for attention output vectors
        