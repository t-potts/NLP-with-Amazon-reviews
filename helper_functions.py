import json
import numpy as np

def import_data(json_path):
    """
    Parameter: String
    Output: 2 Numpy arrays
    
    Reads json from path 'json_path' that has multiple rows of dense matrix and a star rating.
    Processes the file, creates a dense matrix.
    Returns the dense matrix and star rating matrix
    """

    count_vec_list = []
    star_rating_list = []

    vocab_size_saved = False
    with open(json_path, 'r') as f:
        for line in f:
            #creates separate lists for sparse vec data, and star review
            count_vec_list.append(json.loads(line)['count_vec'])
            star_rating_list.append(json.loads(line)['star_rating'])

            #Saves the vocab size on the first pass
            if not vocab_size_saved:
                num_matrix_columns = json.loads(line)['count_vec']['size']
                vocab_size_saved = True


    #creates an empty matrix of the size to fit the read in data
    num_matrix_rows = len(count_vec_list)
    dense_matrix = np.zeros([num_matrix_rows, num_matrix_columns])

    for i, row in enumerate(count_vec_list):
        indices = row['indices']
        values = row['values']

        #Replaces the indices of row i in the dense_matrix with the values
        np.put(dense_matrix[i, :], indices, values)

    #Creates a one-hot encoded good/bad star rating matrix
    y_one_dimensional = np.array(star_rating_list)
    y_array = np.zeros([y_one_dimensional.shape[0], 2])
    y_array[:, 0] = y_one_dimensional
    y_array[:, 1] = y_array[:, 1] + 1
    y_array[:, 1] = y_array[:, 1] - new_y[:, 0]
    
    return dense_matrix, y_array

