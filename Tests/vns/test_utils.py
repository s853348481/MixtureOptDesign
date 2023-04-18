from MixtureOptDesign.vns.utils import generate_initial_design,generate_simplex_lattice_design
import numpy as np
import pytest



@pytest.fixture
def arr():
    design = np.array([[0.        , 0., 1.],
       [0.        , 0.33333333, 0.66666667],
       [0.        , 0.66666667, 0.33333333],
       [0.        , 1.        , 0.        ],
       [0.33333333, 0.        , 0.66666667],
       [0.33333333, 0.33333333, 0.33333333],
       [0.33333333, 0.66666667, 0.        ],
       [0.66666667, 0.        , 0.33333333],
       [0.66666667, 0.33333333, 0.        ],
       [1.        , 0.        , 0.        ]])
        
        
    return design


class TestGenerateSimplexLatticeDesign(object):
    def test_return_shape(self,arr):
        design = generate_simplex_lattice_design(3, 3)
        assert design.shape == (10, 3) and  np.allclose(design,arr)

    def test_return_type(self):
        design = generate_simplex_lattice_design(3, 3)
        assert    isinstance(design, np.ndarray)

    def test_values_within_bounds(self):
        design = generate_simplex_lattice_design(3, 3)
        assert (np.all(design >= 0) and np.all(design <= 1))

    def test_sum_of_points(self):
        design = generate_simplex_lattice_design(3, 3)
        sums = np.sum(design, axis=1)
        assert np.allclose(sums, np.ones_like(sums))

    def test_parameter_validation(self):
        with pytest.raises(ValueError):
            generate_simplex_lattice_design(0, 3)
        with pytest.raises(ValueError):
            generate_simplex_lattice_design(3, 0)
