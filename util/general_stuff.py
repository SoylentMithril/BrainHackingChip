import torch
from exllamav2 import ext
from exllamav2.ext import exllamav2_ext as ext_c

def emb_state_setup(constants, num_attention_heads, num_key_value_heads, head_dim,
                    hidden_states, position_offsets, past_len):
    q_states = hidden_states[1]
    k_states = hidden_states[2]
    v_states = hidden_states[3]
    hidden_states = hidden_states[0]

    offset_tensor = position_offsets if position_offsets is not None else ext.none_tensor
    ext_c.rope_(q_states, constants.sin, constants.cos, past_len, num_attention_heads, head_dim, offset_tensor)
    ext_c.rope_(k_states, constants.sin, constants.cos, past_len, num_key_value_heads, head_dim, offset_tensor)
    return q_states, k_states, v_states

def raw_state_setup(self, direct, constants, cache, loras, hidden_states, position_offsets, batch_size, past_len, q_len):
    q_shape = hidden_states.shape[:-1] + (self.q_proj.out_features,)
    k_shape = hidden_states.shape[:-1] + (self.k_proj.out_features,)
    v_shape = hidden_states.shape[:-1] + (self.v_proj.out_features,)
    q_states = torch.empty(q_shape, device = hidden_states.device, dtype = torch.half)
    batch_keys = None
    batch_values = None
    # If conditions are right we can write the K/V projections directly into the cache

    if direct:
        batch_keys, batch_values = cache.get_kv_state(self.layer_idx, batch_size, 0, past_len)
        k_states = batch_keys.narrow(0, 0, batch_size).narrow(1, past_len, q_len)
        v_states = batch_values.narrow(0, 0, batch_size).narrow(1, past_len, q_len)

    else:
        k_states = torch.empty(k_shape, device = hidden_states.device, dtype = torch.half)
        v_states = torch.empty(v_shape, device = hidden_states.device, dtype = torch.half)

    # RMS norm, Q/K/V projections, position embeddings

    if loras is None or self.temp_lora_size == 0:
        pass_loras = []
        pass_lora_temp = ext.none_tensor
    else:
        pass_loras = [id(x) for x in loras]
        pass_lora_temp = torch.empty((self.temp_lora_size,), dtype = torch.half, device = hidden_states.device)

    if isinstance(past_len, tuple):
        pass_past_len_1 = -1
        pass_past_len_2 = past_len[0]
    elif position_offsets is not None:
        pass_past_len_1 = past_len
        pass_past_len_2 = position_offsets
    else:
        pass_past_len_1 = past_len
        pass_past_len_2 = ext.none_tensor

    ext_c.q_attn_forward_1(self.q_handle,
                            hidden_states,
                            batch_size,
                            q_len,
                            pass_past_len_1,
                            pass_past_len_2,
                            q_states,
                            k_states,
                            v_states,
                            constants.sin,
                            constants.cos,
                            pass_loras,
                            pass_lora_temp)
    return batch_keys, batch_values, q_states, k_states, v_states, pass_loras, pass_lora_temp