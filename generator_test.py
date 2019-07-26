import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import json
from sklearn.metrics import confusion_matrix
from os import listdir


def import_data(json_directory, num_rows):
    """
    Parameter: String, num_rows
    Output: 2 Numpy arrays
    
    Reads jsons in path 'json_directorry' that has multiple rows of dense matrix and a star rating.
    Processes the file, creates a dense matrix.
    Returns the dense matrix and star rating matrix
    """

    count_vec_list_positive = []
    count_vec_list_negative = []
    num_positive, num_negative, file_count = 0, 0, 0
    json_paths = listdir(json_directory)
    path_shuffle = np.arange(len(json_paths))
    np.random.shuffle(path_shuffle)

    while 1:
        
        vocab_size_saved = False
        with open(json_directory + '/' + json_paths[path_shuffle[file_count]], 'r') as f:
            for line in f:
                #creates separate lists for sparse vec data, and star review
                if json.loads(line)['star_rating'] == 1:
                    count_vec_list_positive.append(json.loads(line)['count_vec'])
                else:
                    count_vec_list_negative.append(json.loads(line)['count_vec'])

                #Tracks how many positive/negative reviews have been seen
                if json.loads(line)['star_rating'] == 1:
                    num_positive += 1
                else:
                    num_negative += 1

                #Saves the vocab size on the first pass
                if not vocab_size_saved:
                    num_matrix_columns = json.loads(line)['count_vec']['size']
                    vocab_size_saved = True

                #Builds dense matrix if we have gathered enough of each rating
                if num_positive >= num_rows / 2 and num_negative >= num_rows / 2:
                    num_positive, num_negative = 0, 0
                    #creates an empty matrix of the size to fit the read in data
                    #adds an extra column to hold positive/negative value
                    dense_matrix = np.zeros((num_rows, num_matrix_columns + 1))

                    #randomizes positive indexs
                    positive_shuffle_idx = np.arange(len(count_vec_list_positive))
                    np.random.shuffle(positive_shuffle_idx)

                    #loops through positive reviews to create dense matrix
                    for i in range(len(count_vec_list_positive)):
                        
                        #Breaks loop if half the matrix is filled
                        if i == num_rows / 2: break
                        
                        #selects a random row and gets the indices and values
                        row = count_vec_list_positive[positive_shuffle_idx[i]]
                        indices = row['indices']
                        values = row['values']
                        
                        #Replaces the indices of row i in the dense_matrix with the values
                        np.put(dense_matrix[i, :], indices, values)

                        #sets last column to indicate positive
                        dense_matrix[i, -1] = 1

                    #randomizes negative indexs
                    negative_shuffle_idx = np.arange(len(count_vec_list_positive))
                    np.random.shuffle(negative_shuffle_idx)

                    #loops through negative reviews to create dense matrix
                    for i in range(len(count_vec_list_negative)):
                        #Breaks loop if the matrix is filled
                        if i == num_rows / 2: break  
                        
                        #selects a random row and gets the indices and values
                        row = count_vec_list_positive[negative_shuffle_idx[i]]
                        indices = row['indices']
                        values = row['values']

                        #Replaces the indices of row i in the dense_matrix with the values
                        row_idx = int(i + num_rows / 2)
                        np.put(dense_matrix[row_idx, :], indices, values)

                        #sets last column to indicate negative
                        dense_matrix[int(i + num_rows / 2), -1] = 0
                    
                    #Creates an array of randomized indexs with lengh equal to number of dense_matrix rows
                    shuffle_dense_indexs = np.arange(dense_matrix.shape[0])
                    np.random.shuffle(shuffle_dense_indexs)
                    
                    dense_matrix = dense_matrix[shuffle_dense_indexs]

                    #Declares the lists as empty
                    count_vec_list_positive = []
                    count_vec_list_negative = []

                    yield dense_matrix[:, :-1], dense_matrix[:, ]

            #Tracks the number of files used, and reshuffles the indexes if we have used all the files in the list
            file_count += 1
            if file_count == len(json_paths):
                file_count = 0
                np.random.shuffle(path_shuffle)

for X_, y_ in import_data('jsons', 250):
    print(X_.shape[0])
    print(y_.shape[0])
    break