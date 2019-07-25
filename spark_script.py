from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import udf, concat, col, lit
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import CountVectorizer
from pyspark.sql.types import ArrayType, FloatType, StringType, IntegerType
from pyspark import SparkConf, SparkContext

import sys
from collections import defaultdict
import numpy as np
import re
import time
import pickle

def main():
	sc = SparkContext()
	spark = SparkSession.builder.appName("SimpleApp").getOrCreate()
	print('*'*60, '\n', sc.getConf().getAll(), '\n', '*'*60, '\n') #Prints configuration at start of run on EMR
		
	strip_chars = ".?,!;:\"'()#&"
	rgx = sc.broadcast(re.compile('[%s]' % strip_chars))

	def process_str(row):
		"""
		Input: text row from dataframe
		Output: list of words with punctuation removed

		Note that this must be declared in main for proper function
		"""
		body_list = []
		try:
			for word in row.lower().split(): 
				word = rgx.value.sub('', word)  
				body_list.append(word)
			return body_list
		except Exception as e:
			print(e)
			print(row)
			return ['']

	#Declaration of 'user defined function' so nodes can use them
	process = udf(process_str, ArrayType(StringType()))
	good_bad = udf(good_bad_filter, IntegerType())

	#Directory of reviews: s3://amazon-reviews-pds/tsv/
	#The use of wildcards (*_us_*.gz) allows spark to load all but the non-english reviews
	#full_df = spark.read.csv('s3://amazon-reviews-pds/tsv/*_us_*.gz', sep="\t", header=True, inferSchema=True)
	full_df = spark.read.csv('s3://amazon-reviews-pds/tsv/amazon_reviews_us_Video_DVD_v1_00.tsv.gz', sep="\t", header=True, inferSchema=True)
	
	#Repartitioning the Dataframe allows each task to be split to the workers
	repartition_num = 1000
	full_df = full_df.repartition(repartition_num)

	#Filters out 3 star ratings, and only keeps the review_headline, review_body, and star_rating columns
	#The good_bad function makes 4 and above become 1 (positive review), and 2 and below become 0 (negative review)
	filtered_df = full_df.select('review_headline', 'review_body', 'star_rating')\
	   .filter(full_df.star_rating != 3)\
			.withColumn('star_rating', good_bad('star_rating'))

	#Concatinates the review_headline and review_body columns and renames the column 'text'
	two_col_df = filtered_df.select(concat(col('review_headline'), lit(' '), col('review_body')).alias('text'), filtered_df.star_rating)

	#Turns string into a list of words with the punctuation removed
	text_list_df = two_col_df.withColumn('text_list', process(two_col_df['text']))\
		.select('text_list', 'star_rating')

	#Fitting and transforming the dataset into a count vectorized form
	cv = CountVectorizer(inputCol="text_list", outputCol="count_vec", minDF=1000)
	cv_fit = cv.fit(text_list_df) #need to save vocabulary from this
	cv_transform = cv_fit.transform(text_list_df)
	output_df = cv_transform.select(cv_transform.count_vec, cv_transform.star_rating)

	#Saves the vocabulary and processed dataframe to S3 in JSON format
	vocab = spark.createDataFrame(cv_fit.vocabulary, schema=StringType())
	vocab.coalesce(1).write.mode("overwrite").json('s3://dsi-amazon-neural/vocab')
	output_df.write.mode("overwrite").json('s3://dsi-amazon-neural/jsonfiles')

def good_bad_filter(x):
	"""
	Input: integer
	Output: integer

	Changes from a multistar rating system to good (1) or bad (0)
	"""
	if x >=4: return 1
	else: return 0

if __name__ == "__main__":
	main()