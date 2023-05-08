import numpy as np

from MixtureOptDesign.MNL.mnl_utils import *
# from ..MNL.utils import (
#     compute_cox_direction,
#     generate_beta_params,
#     get_i_optimality_bayesian,
#     get_i_optimality_mnl,
#     get_parameters,
#     transform_varcov_matrix,
#     get_random_initial_design_mnl
# )
from ..HaltonDraws.halton_draws import HaltonDraws


class CoordinateExchangeIOptimal:
    
    """
    This class optimizes a design matrix using the coordinate exchange algorithm.

    Attributes:
        design (np.ndarray): The initial design matrix.
        order (int): The order of the model.
        n_points (int): The number of points to generate in each set.
        iteration (int): The number of iterations to run the algorithm.
        beta (np.ndarray): The beta parameters of the model.
        bayesian (bool): Whether to use Bayesian optimization.
        sigma (np.ndarray): The sigma matrix of the model.
        kappa (int): The kappa parameter of the model.
        num_ingredient (int): The number of ingredient in a mixture
        num_choices (int): The number of choices in each set of the design matrix.
        num_sets (int): The number of alternatives in a choice set.
        i_opt_value (float): The current optimality value of the design matrix.
    """
    
    
    def __init__(
        self,
        num_ingredient:int,
        num_sets:int,
        num_choices:int,
        order:int, 
        n_points: int,
        iteration: int = 10,
        design: np.ndarray=None, 
        beta:np.ndarray=None,
        bayesian:bool=True,
        sigma:np.ndarray=None,
        kappa:int=1
        ) -> None:
        
        """
        Initializes the CoordinateExchangeIOptimal class.

        Args:
            design (np.ndarray): The initial design matrix.
            order (int): The order of the model.
            n_points (int): The number of points to generate in each set.
            iteration (int): The number of iterations to run the algorithm.
            beta (np.ndarray): The beta parameters of the model.
            bayesian (bool): Whether to use Bayesian optimization.
            sigma (np.ndarray): The sigma matrix of the model.
            kappa (int): The kappa parameter of the model.
        """
        
        
        
        design = design
        self._order = order
        self._n_points = n_points
        self._iteration = iteration
        self._num_ingredient, self._num_sets, self._num_choices = num_ingredient,num_sets,num_choices
        self._num_param = sum(get_parameters(self._num_ingredient,self._order)) + 1
        
        
        self._beta = generate_beta_params(self._num_param,self._num_ingredient) if beta is None  else beta
        
        if bayesian:
            
            self.get_i_optimality=get_i_optimality_bayesian
            
            
            if beta is None:
                self.sigma = transform_varcov_matrix(np.identity(self._beta.size + 1),q=self._num_ingredient,k=kappa) if sigma is None else sigma
                self.halt = HaltonDraws(self._beta, self.sigma, 128)
                self._beta = self.halt.generate_draws()
            
        else:
            self.get_i_optimality= get_i_optimality_mnl
        
        
    
    
    def optimize_design(self,desing_numer) -> np.ndarray:
        """
        Optimize design in regards to the optimality criterion.

        Returns:
            np.ndarray: Optimized design 
            
    """
    
        design = get_random_initial_design_mnl(n_ingredients=self._num_ingredient,n_alternatives=self._num_sets,n_choice_sets=self._num_choices,seed=desing_numer[1])

        # set up initial optimality value   
        opt_crit_value_orig =  self.get_i_optimality(design,self._order,self._beta)
        i_best = opt_crit_value_orig 
        i_opt_critc_value = float("inf")
        
        design = design
        it = 0
        for _ in range(self._iteration):
            
            # If there was no improvement in this iteration
            if abs(i_opt_critc_value - i_best) < 0.01:
                break
            
            i_opt_critc_value = i_best
            it += 1
            for j in range(self._num_sets):
                for s in range(self._num_choices):
                    for q in range(self._num_ingredient):
                        cox_directions = compute_cox_direction(design[:, j, s], q, self._n_points)
                        # Loop through each Cox direction
                        for cox_direction in range(cox_directions.shape[0]):
                            # Create candidate design by copying current design
                            canditate_design = design.copy()
                            # Replace the current Mixture with the Cox direction
                            canditate_design[:, j, s] = cox_directions[cox_direction,:]
                            try:
                                # Compute optimality criterion for candidate design
                                i_new_value = self.get_i_optimality(canditate_design,self._order,self._beta)
                                
                            except np.linalg.LinAlgError:
                                # Skip to the next Cox direction if a LinAlgError occurs
                                continue
                            # Update the design and optimality  if there's an improvement
                            if i_new_value > 0 and i_best - i_new_value >= 0.01:
                                design = canditate_design
                                i_best = i_new_value
            
        print(desing_numer[0])   
        print("Original Optimality criterion value: ", opt_crit_value_orig)
        print("Final Optimality criterion value: ", i_best)
        print("Number of iterations: ", it)
        return design.copy()
    

    def get_order(self) -> int:
        return self._order

    def get_beta(self) -> np.ndarray:
        return self._beta

    def get_n_points(self) -> int:
        return self._n_points

    def get_n_iter(self) -> int:
        return self._iteration

   
    
    

    
def unique_rows(design:np.ndarray)->np.ndarray:
        q,j,s = design.shape
        arr = design.T.reshape(j*s,q)
        return np.unique(arr,axis=0)   
