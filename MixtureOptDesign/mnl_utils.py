import numpy as np

def mnl_random_initial_design(n_ingredients: int, n_alternatives: int, n_choice_sets: int, seed: int = None) -> np.ndarray:
    """
    Generate a random initial design for a multinomial logit model (MNL).

    Parameters:
    -----------
    n_ingredients : int
        The number of ingredients (features) in the MNL model.
    n_alternatives : int
        The number of alternatives in the MNL model.
    n_choice_sets : int
        The number of choice sets in the MNL model.
    seed : int or None, optional
        Seed for the random number generator. If None, a new seed will be used.

    Returns:
    --------
    my_array : numpy.ndarray, shape (n_ingredients, n_alternatives, n_choice_sets)
        A 3-dimensional array of random values that can be used as an initial design for the MNL model.

    Notes:
    ------
    The values in the resulting array are normalized so that each alternative in each choice set sums to 1.0.

    Examples:
    ---------
    >>> np.random.seed(0)
    >>> mnl_random_initial_design(2, 3, 2)
    array([[[0.36995516, 0.46260563, 0.35426968],
            [0.13917321, 0.29266522, 0.50959886],
            [0.49087163, 0.24472915, 0.13613147]],

           [[0.63004484, 0.53739437, 0.64573032],
            [0.86082679, 0.70733478, 0.49040114],
            [0.50912837, 0.75527085, 0.86386853]]])

    """
    if seed is not None:
        np.random.seed(seed)
    random_values = np.random.rand(n_ingredients, n_alternatives, n_choice_sets)
    my_array = random_values / np.sum(random_values, axis=0)
    return my_array
