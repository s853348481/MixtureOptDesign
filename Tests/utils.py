import numpy as np
def check_mnl_design_sum(result):
    """
    Check if the sum of ingredient of each mixtures in a MNL random initial design is 1.

    Parameters
    ----------
    result : numpy.ndarray
        The MNL random initial design of shape (n_ingredients, n_alternatives, n_choice_sets).

    Raises
    ------
    ValueError
        If the sum of any ingredients of mixture in the design is not close to 1.

    """
    for i in range(result.shape[2]):
        for j in range(result.shape[1]):
            if not np.isclose(np.sum(result[:, j, i]), 1.0):
                raise ValueError(f"Sum of ingredients in mixture is not close to 1. Resulting values: {result[:, j, i]}")
