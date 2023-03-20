import numpy as np
from ..MNL.mnl_utils import get_i_optimality_mnl,generate_beta_params, get_parameters, compute_cox_direction

class CoordinateExchangeIOptimal:
    def __init__(self, design: np.ndarray, order:int, n_points: int, iteration: int = 10):
        self._design = design.copy()
        self._order = order
        self._n_points = n_points
        self._iteration = iteration
        self._q, self._j, self._s = design.shape
        self._num_param = sum(get_parameters(self._q,self._order)) + 1
        self._beta = generate_beta_params(self._num_param,self._q)
        self._i_opt_value = get_i_optimality_mnl(self._design,self._order,self._beta)
    
    
    def optimize_design(self) -> np.ndarray:
        for _ in range(self._iteration):
            for j in range(self._j):
                for s in range(self._s):
                    for q in range(self._q):
                        cox_directions = compute_cox_direction(self._design[:, j, s], q, self._n_points)
                        for cox_direction in range(cox_directions.shape[0]):
                            canditate_design = np.copy(self._design)
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