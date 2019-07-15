#Natural Language processing using Spark

## What is this project trying to do?

The goal of this project is to analyze customer text from Amazon.com and get insights into what leads to the various star levels of reviews. A predictor will be build to assign a star value to a new review.

## How has this problem been solved before?

NLP on customer reviews is been done many times. An example of this is below: 
https://towardsdatascience.com/detecting-bad-customer-reviews-with-nlp-d8b36134dc7e 

I will be borrowing the fundamental methods of NLP from similar projects. What makes this project unique is that the data consists of 1.7 million+ reviews in a 50+ GB file. Typical NLP packages such as NLTK cannot handle such a large dataset, or if they can the processing will take a prohibitively long time. 

## What is new about this approach?

This project is unique because it will be using a distributed computing network via Apache Spark on Amazon's Web Services to handle a large dataset. I think it will be sucessful because there is a library for Spark specifically for NLP:

https://nlp.johnsnowlabs.com/ 

## Project impact

This project is applivable because there are many datasets that include large volumes of text reviews. Because the project utilizes cloud computing, the cost of running this project in on analysis of text reviews is possible without needing to create a private node network. 

## How will the work be presented?

The primary presentation will be created in a slideshow format. A bokeh based webpage will also be created in order to input text and predict the star rating of that new text document.

## What is the data source?

Amazon provides a large set of reviews dating back to 1996 in an Amazon S3 bucket. A link to this dataset can be found here:
https://registry.opendata.aws/amazon-reviews/

## What are potential problems with this project?

The biggest source of problems lies with implementing Apache Spark. There are many moving parts with the software and debugging is time consuming.

## What needs to be worked on next?

Determining which NLP technique to use in order to gain insight from the large dataset. Subsets of the data can be used in order to use techniques such as NMF, but such techniques may not be feasable or make the best use of the dataset.