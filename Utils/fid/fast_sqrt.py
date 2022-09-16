"""
  Computes trace(sqrt(C1 @ C1.T @ C2 @ C2.T)) 
  If C1.shape=C2.shape=(d, m) it takes O(d m^2) time. 

"""
import torch
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

def trace_of_matrix_sqrt(C1, C2):   
  """
    Computes using the fact that:   eig(A @ B) = eig(B @ A)

    C1, C2    (d, bs) 

    M = C1 @ C1.T @ C2 @ C2.T 

    eig ( C1 @ C1.T @ C2 @ C2.T ) = 
    eig ( C1 @ (C1.T @ C2) @ C2.T ) =      O(d bs^2)
    eig ( C1 @ ((C1.T @ C2) @ C2.T) ) =        O(d bs^2)
    eig ( ((C1.T @ C2) @ C2.T) @ C1 ) =        O(d bs^2)
    eig ( batch_size x batch_size  )      O(bs^3)

  """
  d, bs = C1.shape 
  assert bs <= d, "This algorithm takes O(bs^2d) time instead of O(d^3), so only use it when bs < d.\nGot bs=%i>d=%i. "%(bs, d) # it also computes wrong thing sice it returns bs eigenvalues and there are only d. 
  M = ((C1.t() @ C2) @ C2.t()) @ C1       # computed in O(d bs^2) time.    O(d^^3)
  S = torch.svd( M , compute_uv=True)[1]  # need 'uv' for backprop.
  S = torch.topk(S, bs-1)[0]              # covariance matrix has rank bs-1
  return torch.sum(torch.sqrt(S))


