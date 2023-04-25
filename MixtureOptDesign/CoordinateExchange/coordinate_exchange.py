import numpy as np
from ..MNL.mnl_utils import (
    get_i_optimality_mnl,
    generate_beta_params,
    get_parameters,
    compute_cox_direction
)   

class CoordinateExchangeIOptimal:
    def __init__(
        self, 
        design: np.ndarray, 
        order:int, 
        n_points: int,
        iteration: int = 10,
        beta:np.ndarray=None
        ) -> None:
        self._design = design.copy()
        self._order = order
        self._n_points = n_points
        self._iteration = iteration
        self._q, self._j, self._s = design.shape
        self._num_param = sum(get_parameters(self._q,self._order)) + 1
        
        if beta is None:
            self._beta = generate_beta_params(self._num_param,self._q)
            
        else:
            self._beta = beta
            
        self._i_opt_value = get_i_optimality_mnl(self._design,self._order,self._beta)
    
    
    def optimize_design(self) -> np.ndarray:
        for _ in range(self._iteration):
            for j in range(self._j):
                for s in range(self._s):
                    for q in range(self._q):
                        cox_directions = compute_cox_direction(self._design[:, j, s], q, self._n_points)
                        for cox_direction in range(cox_directions.shape[0]):
                            canditate_design = self._design.copy()
                            canditate_design[:, j, s] = cox_directions[cox_direction,:]
                            i_new_value = get_i_optimality_mnl(canditate_design,self._order,self._beta)
                            if self._i_opt_value >= i_new_value:
                                self._design = canditate_design
                                self._i_opt_value = i_new_value
        return self._design.copy()
    
    def get_design(self) -> np.ndarray:
        return self._design

    def get_order(self) -> int:
        return self._order

    def get_beta(self) -> np.ndarray:
        return self._beta

    def get_n_points(self) -> int:
        return self._n_points

    def get_n_iter(self) -> int:
        return self._n_iter

    def get_i_optimal_design(self) -> np.ndarray:
        return self._i_optimal_design

    def get_i_value_opt(self) -> float:
        return self._i_value_opt
    
    
class ClusteredCoordinateExchangeIOptimal(CoordinateExchangeIOptimal):
    
    def optimize_design(self) -> np.ndarray:
        
        unique_design_points = self.unique_points()
        for _ in range(self._iteration):
            for i in range(len(unique_design_points)):
                
                    for q in range(self._q):
                        cox_directions = compute_cox_direction(unique_design_points[i], q, self._n_points)
                        for cox_direction in range(cox_directions.shape[0]):
                            canditate_design = np.copy(self._design)
                            indices = np.where(np.all(canditate_design == unique_design_points[i].reshape(self._q,1,1), axis=0))
                            subset = canditate_design[:,indices[0],indices[1]]
                            cox = cox_directions[cox_direction,:].reshape(self._q,1,1)
                            subset = np.zeros(subset.shape) + cox
                            i_new_value = get_i_optimality_mnl(canditate_design,self._order,self._beta)
                            if self._i_opt_value >= i_new_value:
                                self._design = canditate_design
                                unique_design_points[i] = cox
                                self._i_opt_value = i_new_value
        return self._design.copy()
    
    def unique_points(self) -> np.ndarray:
        q,j,s = self._design.shape
        design = self._design.copy()
        design = design.reshape(q,1,s*j)
        return np.unique(design,axis=2)
