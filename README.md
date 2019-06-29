# NLP on Amazon reviews
What are common complaints in amazon reviews that lead to a 'bad' review? What are common praises in review that lead to a 'good' review? This project aims to answer those questions

## Approach
I will begin this analysis by using spark to process the data. Because the dataset is so large (250+ million reviews) I will initially start with a subset of the data using only the video games category. Apache spark will then be used to process the raw text of the reviews.

After the text is processed, I will then create a classifier that will predict a star rating based on input text. NLP will also be used to find common words that are linked to each review classification.

## Data sources
Amazon provides a large set of reviews dating back to 1996 in an Amazon S3 bucket. A link to this dataset can be found here:
https://registry.opendata.aws/amazon-reviews/
