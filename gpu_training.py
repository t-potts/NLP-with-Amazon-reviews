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
from helper_functions import import_data, neuralnet_model, batch_generator, count_stars
from sklearn.metrics import confusion_matrix
from os import listdir
import seaborn as sn


def main():
    for X_, y_ in batch_generator('test_data', 40000):
        X_test = X_
        y_test = y_
        break
    print('got training data')
    # this is our input placeholder
    input_img = Input(shape=(X_test.shape[1],))

    # first encoding layer
    ll = Dense(units = 2000, activation = 'relu')(input_img)
    ll = Dense(units = 150, activation = 'relu')(ll)
    #ll = Dense(units = 100, activation = 'relu')(ll)
    #ll = Dense(units = 100, activation='relu')(ll)
    prediction_layer = Dense(units = 1, activation='sigmoid', name='prediction')(ll)
    # this model maps an input to its reconstruction
    model = Model(input_img, prediction_layer)

    # compile model
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

    autoencoder_model_created = False

    if not autoencoder_model_created:

        batch_size = 700
        nb_epoch = 30
        samples_per_epoch = 20

        model.fit_generator(generator=batch_generator('train_data', batch_size),
                            epochs=nb_epoch,
                            steps_per_epoch=samples_per_epoch)


        scores = model.evaluate(X_test, y_test)
        print('Test accuracy = {}'.format(scores[0]))

        y_predictions = model.predict(X_test)
        
        #model.save(model_path)

    else:
        pass
        """model = load_model(model_path)
        scores = model.evaluate(X_test, X_test)
        print('Test mse = {}'.format(scores[0]))"""


    print('*'*50,'y_predictions:', y_predictions[:50])
    confusions = confusion_matrix(y_test, y_predictions > 0.5)
    true_negative = confusions[0,0]
    false_positive = confusions[0,1]
    false_negative = confusions[1, 0]
    true_positive = confusions[1, 1]
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_negative + false_positive)
    print('accuracy:', accuracy)
    print('true_positive:', true_positive)
    print('true_negatives:', true_negative)
    print('false_positive:', false_positive)
    print('false_negative:', false_negative)
    
    precision = true_positive / (true_positive + false_negative)
    recall = true_positive / (true_positive + false_positive)
    f1 = (2 * precision * recall) / (precision + recall)
    print('*'*50)
    print('accuracy: {:0.3f}'.format(accuracy))
    print('Recall: {:0.3f}'.format(true_positive / (true_positive + false_negative)))
    print('Precision: {:0.3f}'.format(true_positive / (true_positive + false_positive)))
    print('f1: {:0.3f}'.format(f1))


    #conf_df = pd.DataFrame([[true_positive, false_positive], [false_negative, true_negative]],                        ['predicted positive', 'predicted negative'],                        ['condition positive', 'condition negative'])




    """    plt.figure(figsize=(15,10))
    sn.set(font_scale=3)
    sn.heatmap(conf_df, annot=True, annot_kws={'size': 26}, fmt='g')"""

if __name__ == "__main__":
    main()
"""
accuracy: 0.9245
true_positive: 4563
true_negatives: 4682
false_positive: 318
false_negative: 437
Recall: 0.91
Precision: 0.93
f1: 0.92
"""

"""
latest test with above code:
accuracy: 0.937475
true_positive: 18639
true_negatives: 18860
false_positive: 1140
false_negative: 1361
**************************************************
accuracy: 0.937
Recall: 0.932
Precision: 0.942
f1: 0.937
"""