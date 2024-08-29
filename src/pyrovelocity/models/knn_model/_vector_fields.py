from beartype import beartype
from torch import Tensor
from typing import Tuple
from typing import List

@beartype
def vector_field_1( t: float,
                    y: Tuple,
                   args: List):
    """
    Vector field of mRNA dynamics of unspliced and spliced counts, based on a regulatory
    function that that takes u,s as input and returns transcription (alpha), splicing (beta)
    and degradation rates (gamma).

    Args:
        t (Float): Integration time. Only used when vector field used in Diffrax library
        and otherwise can be an arbitrary value. 
        y (Tuple): State of the system. Tuple of unspliced (u) and spliced counts (s).
        args (List): List containing a regulatory function that takes u,s as input 
                     and returns transcription (alpha), splicing (beta) and degradation
                     rates (gamma).

    Returns:
        Tuple: Rates of change in y (= unspliced and spliced counts)

    Examples:
        >>> 
    """
    
    u, s = y
    regulatory_function = args[0]
    alpha, beta, gamma = regulatory_function(u,s)
    du = alpha - beta*u
    ds = beta*u - gamma*s
    dy = du, ds 
    
    return dy