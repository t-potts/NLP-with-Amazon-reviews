import json
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, TensorBoard
from keras import metrics
from os import listdir


def neuralnet_model(X_train):
    '''
    defines autoencoder model
    input: X_train (2D np array)
    output: autoencoder (compiled autoencoder model)
    '''
    # this is our input placeholder
    input_img = Input(shape=(X_train.shape[1],))

    # first encoding layer
    encoded1 = Dense(units = 1000, activation = 'relu', name='layer1_256')(input_img)

    """    # second encoding layer
    # note that each layer is multiplied by the layer before
    encoded2 = Dense(units = 200, activation='relu', name='layer2_64')(encoded1)"""

    # first decoding layer
    decoded1 = Dense(units = 1000, activation='relu', name='layer3_256')(encoded1)

    # second decoding layer - this produces the output
    decoded2 = Dense(units = X_train.shape[1], activation='sigmoid', name='layer4_output')(decoded1)
    
    prediction_layer = Dense(units = 1, activation='sigmoid', name='prediction')(decoded2)
    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, prediction_layer)

    # compile model
    autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

    return autoencoder


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
    
    return dense_matrix, np.array(star_rating_list)


def batch_generator(file_dir, batch_size):
    #creates list of filename strings from jsons directory
    dir_list = listdir('jsons')
    
    #Determines number of batches to produce
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    file_counter = 0
    
    #Creates random shuffle index
    file_shuffle_index = np.arange(len(dir_list))
    np.random.shuffle(file_shuffle_index)
    
    #Uses first random shuffle index to select a file
    X, y = import_data('jsons/' + dir_list[file_shuffle_index[file_counter]])
    
    #Creates shuffle index arrays to randomize the data rows as well as the files used
    data_shuffle_index = np.arange(np.shape(X)[0])
    np.random.shuffle(data_shuffle_index)
    
    while 1:
        #Randomizes selection of data rows
        index_batch = data_shuffle_index[batch_size*counter:batch_size*(counter+1)]
        X = X[index_batch, :]
        y = y[index_batch]
        
        #Saves the smallest count of good/bad reviews 
        smallest_good_or_bad_count = min([y[y == 0].shape[0], y[y == 0].shape[0]])
        
        #Creates an array with a 50/50 split of good/bad reviews
        X_batch_good = X[y == 1][0:smallest_good_or_bad_count, :]
        X_batch_bad = X[y == 0][0:smallest_good_or_bad_count, :]
        y_batch_good = y[y == 1][0:smallest_good_or_bad_count]
        y_batch_bad = y[y == 0][0:smallest_good_or_bad_count]
        
        #Stacks the arrays
        X_batch = np.vstack((X_batch_good, X_batch_bad))
        y_batch = np.vstack((y_batch_good.reshape(-1, 1), y_batch_bad.reshape(-1,1)))
        
        #Randomizes the reviews so there isn't a block of good/bad reviews
        batch_shuffle_index = np.arange(X_batch.shape[0])
        np.random.shuffle(batch_shuffle_index)
        X_batch = X_batch[batch_shuffle_index, :]
        y_batch = y_batch[batch_shuffle_index]
        
        counter += 1
        
        """if autoencoder_layer:
            yield X_batch, X_batch
        else:"""
        yield X_batch, y_batch
        
        
        if (counter >= number_of_batches):
            counter=0
            file_counter +=1
            
            if file_counter == len(dir_list):
                np.random.shuffle(file_shuffle_index)
                file_count = 0
                
            #loads next file and resets shuffle index
            X, y = import_data('jsons/' + dir_list[file_shuffle_index[file_counter]])
            data_shuffle_index = np.arange(np.shape(X)[0])
            np.random.shuffle(data_shuffle_index)

def batch_generator_old(file_dir, batch_size, samples_per_epoch):
    #creates list of filename strings from jsons directory
    dir_list = listdir('jsons')
    
    #Determines number of batches to produce
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    file_counter = 0
    
    #Creates random shuffle index
    file_shuffle_index = np.arange(len(dir_list))
    np.random.shuffle(file_shuffle_index)
    
    #Uses first random shuffle index to select a file
    X, y = import_data('jsons/' + dir_list[file_shuffle_index[file_counter]])
    
    #Creates shuffle index arrays to randomize the data rows as well as the files used
    data_shuffle_index = np.arange(np.shape(X)[0])
    np.random.shuffle(data_shuffle_index)
    
    while 1:
        #Randomizes selection of data rows
        index_batch = data_shuffle_index[batch_size*counter:batch_size*(counter+1)]
        X = X[index_batch, :]
        y = y[index_batch]
        
        #Saves the smallest count of good/bad reviews 
        smallest_good_or_bad_count = min([y[y == 0].shape[0], y[y == 0].shape[0]])
        
        #Creates an array with a 50/50 split of good/bad reviews
        X_batch_good = X[y == 1][0:smallest_good_or_bad_count, :]
        X_batch_bad = X[y == 0][0:smallest_good_or_bad_count, :]
        y_batch_good = y[y == 1][0:smallest_good_or_bad_count]
        y_batch_bad = y[y == 0][0:smallest_good_or_bad_count]
        
        #Stacks the arrays
        X_batch = np.vstack((X_batch_good, X_batch_bad))
        y_batch = np.vstack((y_batch_good.reshape(-1, 1), y_batch_bad.reshape(-1,1)))
        
        #Randomizes the reviews so there isn't a block of good/bad reviews
        batch_shuffle_index = np.arange(X_batch.shape[0])
        np.random.shuffle(batch_shuffle_index)
        X_batch = X_batch[batch_shuffle_index, :]
        y_batch = y_batch[batch_shuffle_index]
        
        counter += 1
        
        y_out = np.zeros([y_batch.shape[0], 2])
        y_out[:, 0] = y_batch.reshape(-1,)
        y_out[:, 1] = y_out[:, 1] + 1
        y_out[:, 1] = y_out[:, 1] - y_out[:, 0]
       
        yield X_batch, y_out
        
        if (counter >= number_of_batches):
            counter=0
            file_counter +=1
            
            if file_counter == len(dir_list):
                np.random.shuffle(file_shuffle_index)
                file_count = 0
                
            #loads next file and resets shuffle index
            X, y = import_data('jsons/' + dir_list[file_shuffle_index[file_counter]])
            data_shuffle_index = np.arange(np.shape(X)[0])
            np.random.shuffle(data_shuffle_index)