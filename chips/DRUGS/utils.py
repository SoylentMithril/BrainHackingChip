#As explained in https://github.com/EGjoni/DRUGS/blob/main/porting/A%20Guide%20to%20Making%20DRUGS.md 
import torch

def get_slice(input_tensor, slice_dim, start_idx, end_idx):
  slices = [slice(None)] * input_tensor.ndim
  slices[slice_dim] = slice(start_idx, end_idx)
  return slices

def get_perturbed_vectors(input_vectors, max_theta_radians):
    r"""
        middleschool trig rotation in n-dimensional space
        max_theta_radians : defines the maximum angle in radians to rotate input vectors by in a random direction.
    """
    component_slice = get_slice(input_vectors, input_vectors.ndim-1, 0, 1)
    random_angles = torch.rand_like(input_vectors[component_slice].squeeze(-1), device=input_vectors.device, dtype=input_vectors.dtype) * max_theta_radians
    #The next three lines are an old family recipe for cooking up orthogonal vectors.
    random_vectors = torch.rand_like(input_vectors)
    projections = input_vectors * (torch.sum(random_vectors * input_vectors, dim=-1, keepdim=True) / 
                   torch.sum(input_vectors * input_vectors, dim=-1, keepdim=True))
    orthoshifts = random_vectors - projections
    #Delicious. Now sprinkle generously and knead them back into our base vectors
    target_shifts = (torch.norm(input_vectors, dim=-1, keepdim=True) 
                    * (orthoshifts / torch.norm(orthoshifts, dim=-1, keepdim=True)))
    results = (input_vectors * torch.cos(random_angles).unsqueeze(-1)) + (target_shifts * torch.sin(random_angles).unsqueeze(-1))
    return results

def verifyAverage(dvalues, values, max_theta):
    dvalues64 = dvalues.to(dtype=torch.float64)
    values64 = values.to(dtype=torch.float64)
    dvalues_norm = torch.nn.functional.normalize(dvalues64, p=2, dim=-1)
    values_norm = torch.nn.functional.normalize(values64, p=2, dim=-1)
    dot_products = (dvalues_norm * values_norm).sum(dim=-1) 
    angles = torch.acos(torch.clamp(dot_products, -1, 1)) 
    result_angle = angles.mean().item()
    result = f"average angle is {round(result_angle, 5)}, which is ~ {round(result_angle/(max_theta/2), 2)}X of expectation"
    #print(result)
    return result