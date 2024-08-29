import torch
import torch.nn.functional as F
from beartype import beartype
from torch import Tensor

@beartype
def regulatory_function_1(u: Tensor,
                          s: Tensor,
                          h1: int = 100,
                          h2: int = 100):
    """
    Regulatory function that contains a neural net with two hidden layers that takes unspliced and spliced
    counts as input and returns transcription (alpha), splicing (beta) and degradation rates (gamma).

    Args:
        u (Tensor): Unspliced counts
        s (Tensor): Spliced counts
        h1 (int): Nodes in hidden layer 1
        h2 (int): Nodes in hidden layer 2

    Returns:
        Tuple: transcription (alpha), splicing (beta) and degradation rate (gamma).

    Examples:
        >>> 
    """
    
    input = torch.tensor(np.array([np.array(u), np.array(s)]).T)
    
    l1 = torch.nn.Linear(2, h1)
    l2 = torch.nn.Linear(h1, h2)
    l3 = torch.nn.Linear(h2, 3)
    
    x = l1(input)
    x = F.leaky_relu(x)
    x = l2(x)
    x = F.leaky_relu(x)
    x = l3(x)
    
    output = torch.sigmoid(x)
    beta = output[:,0]
    gamma = output[:,1]
    alphas = output[:,2]
    
    return alphas, beta, gamma