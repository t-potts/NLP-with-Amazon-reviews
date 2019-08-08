# Neural Predictor using NLP on Amazon Reviews
This project built a neural network to classify a text review as positive or negative without the need for an existing rating system. This allows a business to quickly quantify customer reactions to products without the need for a human to manually read the text reviews. 

## Approach
There are two main parts to this project. 

First I extracted the data from S3 and processed it into a count vectorized format using Apache Spark and a 21 node Spark cluster. The resulting data was saved back to S3. 

The second part was to use the processed data and train a neural net using Keras/Tensorflow. This was accomplished by using a GPU EC2 instance because the neural net was too large to be trained on a local machine. Once the model was trained it was saved and downloaded locally where predictions could be made.

## Exploratory Data Analysis
That main purpose of EDA was to understand the distribution of the ratings. Upon taking a 2000 review sample, it became very clear that there was a disproportionate number of 5 star reviews. I also made an initial neural network based on a 5 star classification output and found that the network struggled to differentiate between star ratings on the low end, and star ratings on the high end. My hypothesis is that people often use similar language for 4/5 and 1/2 star ratings and the actual star they assign is dependent on the individual users preference.

To overcome user dependent ratings, I decided to classify 4/5 stars as a 'good' reviews, and 1/2 stars as a 'bad' review. I filtered out the 3 star reviews because I believed an overall sentiment of the product being good or bad was too ambiguous. This ultimately proved very effective when it came to classifying unseen reviews as good or bad.

To overcome the problem of inbalanced classes I realized I built a training data generator that would randomly select an equal number of good and bad reviews to train the network.

I also split the data into an 80/20 train-test split for validating the results of the model.

## Preprocessing with spark
Preprocessing consisted of several steps. 

First the data was read into a Spark dataframe and repartitioned into 1000 parts because there were over 150+ million reviews to process. The repartitioning was critical to not only allow the cluster to be utilized efficiently, but also to not overload the 12 GB RAM allotment for each worker node.

Next I filtered out the 3 star reviews. I classified 1/2 stars as 0, and 4/5 stars as 1. This would allow me to use a sigmoid function at the end of the my neural network for prediction.

After this I combined the columns of review_headline and review_body into a single column I called text. I then used a user defined function (a type of function unique to Spark) which allowed each node to apply the function to a single review. The function removed all characters, and split the string a list of individual words.

After the raw text is in this format, we need to perform the most computationally expensive part of the preprocessing which is count vectorization. This consists of assigning every unique word to a number, with the caveat that a word has to appear at least 1000 times in all of the text reviews to be included. The list of words is the converted into a dictionary where the key is the word number in the vocabulary list, and the value corresponds to the number of times that word appears in the review.

Finally all of the reviews are processed and are then saved in JSON format to S3. Because the dataframe was repartitioned into 1000 parts, the output is saved into 1000 different files. Each JSON file was roughly 50 MB for an estimated total of 50 GB for the entire dataset.

## Neural network training
The most difficult part of training the network was getting the processed data into a format that the neural net could train on. To handle this I created a generator that would accept a directory and the number of reviews to return per yield. This was necessarry because each review was fed to the network as a 78,133 column numpy array and took ~0.5 KB of space. This means that a training batch of 2000 would take a full GB of RAM. The generator also randomize which of the 1000 files was selected, as well as the order that the reviews were sequentially pulled from the file. Once the network was trained it was saved locally to be used for future predictions.

The structure of the network consisted of 78,133 input nodes, one for each word in the vocabulary, two hidden layers of 2000 and 150 nodes respectively,and a single output node for prediction. This resulted in network that could predict on the test set with 94% accuracy and an f1 score of 0.96

## Next steps
Since the model is created and functional, the next step is to make it easily deployable. The model could be set up as an API call so that inputting a collection of text reviews would return the percentage of reviews that are positive.

I would also like to explore other neural network designs, such as RNNs. Because I utilized a bag-of-words approach, the ordering of the words was not considered when training the network.

## Acknowledgement
Special thanks to Joseph Gartner and Dan Rupp for mentoring and guiding me throughout the duration of this project.

## Data sources
Amazon provides a large set of reviews dating back to 1996 in an Amazon S3 bucket. A link to this dataset can be found here:
https://registry.opendata.aws/amazon-reviews/
