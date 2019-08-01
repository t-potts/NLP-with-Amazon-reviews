import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
import re
from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, TensorBoard
from keras import metrics
import json
from helper_functions import import_data, neuralnet_model, batch_generator, count_stars, test_creator
from sklearn.metrics import confusion_matrix
from os import listdir



def main():
    X_test, y_test = test_creator('test_data', 1)
    print('got training data')
    # this is our input placeholder
    input_img = Input(shape=(X_test.shape[1],))

    # first dense layer of 2000 nodes
    ll = Dense(units = 2000, activation = 'relu')(input_img)
    # second dense layer of 150 nodes
    ll = Dense(units = 150, activation = 'relu')(ll)
    # single prediction node
    prediction_layer = Dense(units = 1, activation='sigmoid', name='prediction')(ll)
    
    model = Model(input_img, prediction_layer)

    
    # Set to True if there is an existing model saved to 'models/trained_model'
    autoencoder_model_created = False

    if not autoencoder_model_created:
        # compile model
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

        # designate batch size, number of epochs, and samples per epoch
        batch_size = 2000
        nb_epoch = 15
        samples_per_epoch = 20

        # batch generator provides batch_size number of samples at a time to minimize RAM usage
        model.fit_generator(
            generator=batch_generator('train_data', batch_size),
            epochs=nb_epoch,
            steps_per_epoch=samples_per_epoch)

        # Creates a test set from the test_data directory and runs the test
        X_test, y_test = test_creator('test_data', 50000)
        y_predictions = model.predict(X_test)
        
        # Saves the trained model
        model.save('models/trained_model')

    else:
        # Loads in the trained model
        model = load_model('models/trained_model')

        # Creates a test set from the test_data directory and runs the test
        X_test, y_test = test_creator('test_data', 50000)
        y_predictions = model.predict(X_test)


    
    # Defines various statistics
    confusions = confusion_matrix(y_test, y_predictions > 0.5)
    true_negative = confusions[0,0]
    false_positive = confusions[0,1]
    false_negative = confusions[1, 0]
    true_positive = confusions[1, 1]
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_negative + false_positive)
    precision = true_positive / (true_positive + false_negative)
    recall = true_positive / (true_positive + false_positive)
    f1 = (2 * precision * recall) / (precision + recall)

    # Prints out various stats based on the test predictions
    print('*'*50, '\n', 'positive is being predicted')
    print('accuracy:', accuracy)
    print('true_positive:', true_positive)
    print('true_negatives:', true_negative)
    print('false_positive:', false_positive)
    print('false_negative:', false_negative)
    

    print('*'*50)
    print('accuracy: {:0.3f}'.format(accuracy))
    print('Recall: {:0.3f}'.format(true_positive / (true_positive + false_negative)))
    print('Precision: {:0.3f}'.format(true_positive / (true_positive + false_positive)))
    print('f1: {:0.3f}'.format(f1))

if __name__ == "__main__":
    main()
