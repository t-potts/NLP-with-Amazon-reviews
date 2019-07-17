#Natural Language processing using Spark

## What is this project trying to do?

The goal of this project is to analyze customer text from Amazon.com and get insights into what leads to the various star levels of reviews. A distributed computing network will be used to initially process the data. This processed data will then be used to train a neural network that can predict the star rating of new text reviews.

## How has this problem been solved before?

When it comes to processing text, dimensionality reduction is a commonly used technique. An example of this can be seen here: 
https://towardsdatascience.com/dimensionality-reduction-for-machine-learning-80a46c2ebb7e

Neural networks can also be used to make predictions as seen here: https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/

I will be borrowing from both of these techniques. I will train a network to reduce dimensionality, and then use the trained weights from all but the last prediction layer as a basis for creating a new neural network utilizing transfer learning.

## What is new about this approach?

There are two things that make this project unique. First is the large dataset, which consists of 1.7 million+ reviews in a 50+ GB file. The second is the use of a transfer learning based neural net to make predictions. This methodology will make use of the large training set that the dataset provides

https://nlp.johnsnowlabs.com/ 

## Project impact

This project is applicable because there are many datasets that include large volumes of text reviews. Because the project utilizes cloud computing, the cost of running this project in on analysis of text reviews is possible without needing to create a private node network.  

## How will the work be presented?

The primary presentation will be created in a slideshow format. A web page will be created in order to submit a review and have a star rating predicted for it. 

## What is the data source?

Amazon provides a large set of reviews dating back to 1996 in an Amazon S3 bucket. A link to this dataset can be found here:
https://registry.opendata.aws/amazon-reviews/

## What are potential problems with this project?

The biggest source of problems lies the technical issues associated with the technology that can lead to hours long troubleshooting sessions. To alleviate this, I will use the pamadoro technique. If after a 25 minute session I have made no progress towards a solution, I will re-evaluate my approach and consider asking my for guidance. 

## What needs to be worked on next?

Getting the process saved into an S3 bucket. EDA also needs to be performed. Once that is finished, I will begin to construct the Neural network. I will then search for a pretrained autoencoding neural network for text. I can utilize the weights of all but the last layer and freeze those weights. I will add some new hidden layers that will then be trained on my data to predict the start rating.