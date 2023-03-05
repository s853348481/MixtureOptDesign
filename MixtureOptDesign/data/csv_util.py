import numpy as np
import csv

def read_csv_file(filename):
    """
    Read a CSV file and return a 3D numpy array of alternative data.

    Parameters
    ----------
    filename : str
        The path to the CSV file.

    Returns
    -------
    numpy.ndarray
        A 3D numpy array with shape (num_attributes, num_alternatives, num_choice_sets),
        where num_attributes is the number of columns in the CSV file minus 1 (the last
        column is assumed to be the choice set identifier), num_alternatives is the number
        of rows divided by the number of unique choice set identifiers, and num_choice_sets
        is the number of unique choice set identifiers. The values in the array are the
        attribute values for each alternative.

    Raises
    ------
    ValueError
        If the CSV file does not have the expected format or contains invalid data.

    Example
    -------
    >>> data = read_csv_file('alternatives.csv')
    >>> print(data.shape)
    (16, 2, 3)

    """

    # Read csv file using np.genfromtxt()
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)

    # Extract information
    unique_choice_sets = np.unique(data[:, 3])
    num_choice_sets = len(unique_choice_sets)
    num_alternatives = data.shape[0] // num_choice_sets

    # Reshape and transpose data
    data = data[:, :-1].reshape((num_choice_sets, num_alternatives, -1)).transpose(2, 1, 0)

    return data

def write_csv_file(filename, data):
    """Write a CSV file from a 3-dimensional numpy array.

    Parameters
    ----------
    filename : str
        The name of the CSV file to write.
    data : numpy.ndarray
        A 3-dimensional numpy array containing the data to write.
        The array should have shape (num_attributes, num_alternatives, num_choice_sets).

    Returns
    -------
    None

    Notes
    -----
    The CSV file will have a header row with the names of the attributes,
    and an additional column for the choice set.
    The choice set values are integers starting from 1.

    Examples
    --------
    >>> data = np.array([[[1, 2, 3], [4, 5, 6]],
                         [[7, 8, 9], [10, 11, 12]],
                         [[13, 14, 15], [16, 17, 18]]])
    >>> write_csv_file('output.csv', data)

    The resulting CSV file will have the following contents:

    ingredients_1,ingredients_2,ingredients_3,choice_set
    1,2,3,1
    4,5,6,1
    7,8,9,2
    10,11,12,2
    13,14,15,3
    16,17,18,3
    """
    
    # Extract information
    num_attributes, num_alternatives, num_choice_sets = data.shape
    unique_choice_sets = np.arange(num_choice_sets) + 1
    
    # Reshape and transpose data
    data = data.transpose(2, 1, 0).reshape(-1, num_attributes )
    choice_sets = np.repeat(unique_choice_sets, num_alternatives)

    # Add choice set column and write to CSV file
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([f'ingredients_{i+1}' for i in range(num_attributes)] + ['choice_set'])
        for i, row in enumerate(data):
            csvwriter.writerow( row.tolist() + [choice_sets[i]])



