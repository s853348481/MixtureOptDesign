import numpy as np
from ..mnl_utils import get_i_optimality_mnl,generate_beta_params, get_parameters

class CoordinateExchangeIOptimal:
    def __init__(self, design: np.ndarray, order:int, n_points: int, n_iter: int = 10):
        self.__design = design.copy()
        self.__order = order
        self.__n_points = n_points
        self.__n_iter = n_iter
        self.__q = design.shape[1]
        self.__num_param = sum(get_parameters(self.__q,self.__order))
        self.__beta = generate_beta_params(self.__num_param,self.q)
        self.__i_value_opt = get_i_optimality_mnl(self.__design,self.__order,self.__beta)
    
    
    def optimize_design(self) -> np.ndarray:
        
        pass