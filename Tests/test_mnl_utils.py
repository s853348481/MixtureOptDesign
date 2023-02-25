import numpy as np
import pytest

from MixtureOptDesign.mnl_utils import *
from Tests.utils import check_mnl_design_sum

# test case to check that the function returns an array with the correct shape
def test_mnl_random_initial_design_shape() -> None:
    n_ingredients = 3
    n_alternatives = 4
    n_choice_sets = 5
    result = mnl_random_initial_design(n_ingredients, n_alternatives, n_choice_sets)
    assert result.shape == (n_ingredients, n_alternatives, n_choice_sets), f"Resulting shape is {result.shape}, expected {(n_ingredients, n_alternatives, n_choice_sets)}"

# test case to check that the function returns an array where each alternative's values sum to 1
def test_mnl_random_initial_design_sum() -> None :
    n_ingredients = 3
    n_alternatives = 4
    n_choice_sets = 5
    result = mnl_random_initial_design(n_ingredients, n_alternatives, n_choice_sets)
    check_mnl_design_sum(result)

# test case to check that the function returns different results when using different random seeds
def test_mnl_random_initial_design_seed() -> None:
    n_ingredients = 3
    n_alternatives = 4
    n_choice_sets = 5
    result1 = mnl_random_initial_design(n_ingredients, n_alternatives, n_choice_sets, seed=123)
    result2 = mnl_random_initial_design(n_ingredients, n_alternatives, n_choice_sets, seed=456)
    assert not np.allclose(result1, result2)
