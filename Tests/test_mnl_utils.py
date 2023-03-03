import numpy as np
import pytest

from MixtureOptDesign.mnl_utils import *
from Tests.utils import check_mnl_design_sum


@pytest.fixture
def arr():
    design = np.array([
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0, 0],
            ],
            [
                [0.2, 0.3, 0.4],
                [0.5, 0.5, 0.2],
                [0.1, 0, 1.0],
            ],
            [
                [0.7, 0.5, 0.3],
                [0.1, 0, 0.2],
                [0.2, 1.0, 0],
            ],
        ])
        
        
    return design   
      

class TestInitialDesign(object):
    
    # test case to check that the function returns an array with the correct shape
    def test_design_shape(self) -> None:
        n_ingredients = 3
        n_alternatives = 4
        n_choice_sets = 5
        result = get_random_initial_design_mnl(n_ingredients, n_alternatives, n_choice_sets)
        assert result.shape == (n_ingredients, n_alternatives, n_choice_sets), f"Resulting shape is {result.shape}, expected {(n_ingredients, n_alternatives, n_choice_sets)}"

    # test case to check that the function returns an array where each alternative's values sum to 1
    def test_sum(self) -> None :
        n_ingredients = 3
        n_alternatives = 4
        n_choice_sets = 5
        result = get_random_initial_design_mnl(n_ingredients, n_alternatives, n_choice_sets)
        check_mnl_design_sum(result)

    # test case to check that the function returns different results when using different random seeds
    def test_seed(self) -> None:
        n_ingredients = 3
        n_alternatives = 4
        n_choice_sets = 5
        result1 = get_random_initial_design_mnl(n_ingredients, n_alternatives, n_choice_sets, seed=123)
        result2 = get_random_initial_design_mnl(n_ingredients, n_alternatives, n_choice_sets, seed=456)
        assert not np.allclose(result1, result2)

class TestGetParameters(object):
    
    def test_order_1(self)->None:

        # # Test case for order 1 and q=3
        q = 3
        order = 1
        expected_result = (q -1,0 ,0)
        assert get_parameters(q, order) == expected_result
    
    def test_order_2(self)->None:
        # # Test case for order 2 and q=4   
        q = 4
        order = 2
        expected_result = (q-1,q * (q - 1)/2, 0)
        assert get_parameters(q, order) == expected_result
    
    def test_order_3(self)->None:
        # # Test case for order 3 and q=5
        q = 5
        order = 3
        expected_result = (q-1, q * (q - 1)/2, q * (q - 1) * (q - 2)/6)
        assert get_parameters(q, order) == expected_result
    
class TestGetBetaCoefficients(object):
    
    def test_linear(self):
        beta = np.arange(1,4)
        q = 4
        order = 1
        expected_output = (np.arange(1,4), np.empty((0,)), np.empty((0,)))
        assert all(np.array_equal(x, y) for x, y in zip(get_beta_coefficients(beta, q, order), expected_output))
        
        
    def test_quadratic(self):
        beta = np.arange(1,10)
        q = 4
        order = 2
        expected_output = (np.arange(1,4), np.arange(4,10), np.empty((0,)))
        assert all(np.array_equal(x, y) for x, y in zip(get_beta_coefficients(beta, q, order), expected_output))


    def test_cubic(self):
        beta = np.arange(1,7)
        q = 3
        order = 3
        expected_output = (np.arange(1,3), np.arange(3,6), np.arange(6,7))
        # Test for equality
        assert all(np.array_equal(x, y) for x, y in zip(get_beta_coefficients(beta, q, order), expected_output))


    def test_order_error(self):
        beta = np.random.rand(10)
        q = 4
        order = 4
        with pytest.raises(ValueError):
            get_beta_coefficients(beta, q, order)
            
    def test_beta_error(self):
        beta = np.array([1, 2, 3, 4, 5, 6])
        q = 4
        order = 2
        with pytest.raises(ValueError):
            get_beta_coefficients(beta, q, order)

class TestMultiplyArrays(object):


    def test_multiply_arrays(self):
    
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        c = np.array([7, 8, 9])
        expected_result = np.array([28, 80, 162])
        assert np.array_equal(multiply_arrays(a, b, c), expected_result)

    def test_multiply_arrays_empty_input(self):
        with pytest.raises(ValueError):
            multiply_arrays()

        
class TestInteractionTerms(object):
    
    
    def test_2_order_interaction(self,arr)-> None:
        # Test with interaction = 2
        
    
        axis0_axis1 = np.multiply(arr[0], arr[1])
        axis0_axis2 = np.multiply(arr[0], arr[2])
        axis1_axis2 = np.multiply(arr[1], arr[2])
        interaction = np.stack((axis0_axis1, axis0_axis2, axis1_axis2), axis=0)
        assert np.array_equal(interaction,interaction_terms(arr,2))


    def test_3_order_interaction(self,arr)-> None:
        # Test with interaction = 3
        
        
        axis0_axis1_axis2 = arr[0] * arr[1] * arr[2]
        interaction = axis0_axis1_axis2.reshape(1,3,3)
        assert np.array_equal(interaction,interaction_terms(arr,3))


    def test_interaction_non_int(self,arr):
        # Test with invalid input: non-integer interaction
        arr = np.random.rand(3, 4)
        with pytest.raises(TypeError):
            interaction_terms(arr, 2.5)
        
    def test_empty_array(self,arr):
        # Test with invalid input: empty array
        arr = np.array([])
        with pytest.raises(ValueError):
            interaction_terms(arr, 2)
            
class TestModelMatrix(object):
    
    def test_order_1(self,arr):
        # Test order 1 with small design matrix
        
        expected_model = np.array([[[0.1, 0.2, 0.3],
                                    [0.4, 0.5, 0.6],
                                    [0.7, 0. , 0. ]],

                                    [[0.2, 0.3, 0.4],
                                    [0.5, 0.5, 0.2],
                                    [0.1, 0. , 1. ]]])
        
        result_model = get_model_matrix(arr, 1)
        assert np.array_equal(result_model, expected_model)

    def test_order_2(self,arr):
        # Test order 2 with small design matrix
        expected_model = np.array([[[0.1 , 0.2 , 0.3 ],
                                    [0.4 , 0.5 , 0.6 ],
                                    [0.7 , 0.  , 0.  ]],

                                [[0.2 , 0.3 , 0.4 ],
                                    [0.5 , 0.5 , 0.2 ],
                                    [0.1 , 0.  , 1.  ]],

                                [[0.02, 0.06, 0.12],
                                    [0.2 , 0.25, 0.12],
                                    [0.07, 0.  , 0.  ]],

                                [[0.07, 0.1 , 0.09],
                                    [0.04, 0.  , 0.12],
                                    [0.14, 0.  , 0.  ]],

                                [[0.14, 0.15, 0.12],
                                    [0.05, 0.  , 0.04],
                                    [0.02, 0.  , 0.  ]]])
        result_model = get_model_matrix(arr, 2)
        assert np.allclose(result_model,expected_model)
        #np.testing.assert_array_almost_equal(result_model, expected_model)
        

    def test_order_3(self,arr):
        # Test order 3 with small design matrix
        expected_model = np.array([[[0.1  , 0.2  , 0.3  ],
                                    [0.4  , 0.5  , 0.6  ],
                                    [0.7  , 0.   , 0.   ]],

                                [[0.2  , 0.3  , 0.4  ],
                                    [0.5  , 0.5  , 0.2  ],
                                    [0.1  , 0.   , 1.   ]],

                                [[0.02 , 0.06 , 0.12 ],
                                    [0.2  , 0.25 , 0.12 ],
                                    [0.07 , 0.   , 0.   ]],

                                [[0.07 , 0.1  , 0.09 ],
                                    [0.04 , 0.   , 0.12 ],
                                    [0.14 , 0.   , 0.   ]],

                                [[0.14 , 0.15 , 0.12 ],
                                    [0.05 , 0.   , 0.04 ],
                                    [0.02 , 0.   , 0.   ]],

                                [[0.014, 0.03 , 0.036],
                                    [0.02 , 0.   , 0.024],
                                    [0.014, 0.   , 0.   ]]])
        
        
        result_model = get_model_matrix(arr, 3)
        assert np.allclose(result_model, expected_model)
        #np.testing.assert_array_almost_equal

    def test_order_out_of_bounds(self,arr):
        
        with pytest.raises(ValueError):
            get_model_matrix(arr, 4)
        
class TestGetUtilities(object):
    
    def test_order_1(self,arr):

            
            beta_star = np.array([1.0, 2.0])
            beta_2FI = np.array([3.0,4.0, 5.0])
            beta_3FI = np.array([6.0])
            order = 1
            
            expected_output = np.array([[0.5, 0.8, 1.1],
                                        [1.4, 1.5, 1. ],
                                        [0.9, 0. , 2. ]])
            
            assert np.allclose(get_utilities(arr, beta_star, beta_2FI, beta_3FI, order), expected_output)


    def test_order_2(self,arr):
        
            
            beta_star = np.array([1.0, 2.0])
            beta_2FI = np.array([3.0,4.0, 5.0])
            beta_3FI = np.array([6.0])
            order = 2
            
            expected_output = np.array([[1.54, 2.13, 2.42],
                                        [2.41, 2.25, 2.04],
                                        [1.77, 0.  , 2.  ]])
            
            assert np.allclose(get_utilities(arr, beta_star, beta_2FI, beta_3FI, order), expected_output)
    def test_order_3(self,arr):
        
            
            beta_star = np.array([1.0, 2.0])
            beta_2FI = np.array([3.0,4.0, 5.0])
            beta_3FI = np.array([6.0])
            order = 3
            
            expected_output = np.array([[1.708, 2.49 , 2.852],
                                        [2.65 , 2.25 , 2.328],
                                        [1.938, 0.   , 2.   ]])
            
            assert np.allclose(get_utilities(arr, beta_star, beta_2FI, beta_3FI, order), expected_output)
            
class TestGenerateBetaParam(object):
    def test_1D_array(self):
        
        # Test that the function returns a 1-D numpy array
        beta = generate_beta_params(5, 3)
        assert isinstance(beta, np.ndarray)
        assert beta.ndim == 1
        
    
    def test_length_array(self):
        
        # Test that the length of the returned array is correct
        beta = generate_beta_params(5, 3)
        assert len(beta) == 4
        
    
    def test_value_error(self):
        
        # Test that the function raises a ValueError if q is less than 1 or greater than num_params
        with pytest.raises(ValueError):
            generate_beta_params(5, 0)
        with pytest.raises(ValueError):
            generate_beta_params(5, 6)


class TestComputeCoxDirection(object):
    
    def test_sum_to_one(self):
        q = np.array([0.3,0.5,0.2])
        cox = compute_cox_direction(q,0)
        
        assert np.allclose(np.sum(cox,axis=1),np.ones(30))
        

class TestGetChoiceProbabilities(object):
    

    def test_2D_array(self):
        # Test that the function returns a 2-D numpy array
        U = np.array([[1, 2, 3], [4, 5, 6]])
        P = get_choice_probabilities(U)
        assert isinstance(P, np.ndarray)
        assert P.ndim == 2
    
    
    def test_shape(self):
        # Test that the shape of the returned array is correct
        U = np.array([[1, 2, 3], [4, 5, 6]])
        P = get_choice_probabilities(U)
        assert P.shape == (2, 3)
    
    
    def test_sum_to_one(self):
        # Test that the sum of the probabilities for each decision is 1
        U = np.array([[1, 2, 3], [4, 5, 6]])
        P = get_choice_probabilities(U)
        assert np.allclose(np.sum(P, axis=1), np.ones(2))
    
    def test_calculate(self):
        # Test that the probabilities are calculated correctly
        U = np.array([[1, 2, 3], [4, 5, 6]])
        P = get_choice_probabilities(U)
        assert np.allclose(P, np.array([[0.09003057, 0.24472847, 0.66524096], [0.09003057, 0.24472847, 0.66524096]]))
    
    
