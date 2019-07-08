import nltk
nltk.download('stopwords')
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import udf
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.ml import Pipeline
from pyspark.sql.types import ArrayType, FloatType, StringType
from nltk.stem.porter import PorterStemmer
from pyspark import SparkConf, SparkContext
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import sys
from collections import defaultdict
import numpy as np
import re
import time


def main():

	def process_str(row):
		"""
		Parameters
		----------
		Row: string:
		  Row of dataframe that contains a string doc

		Returns
		----------
		List of strings:
		A stemmed list of words with common stopwords removed

		Notes: function must be declared in main function because it utilizes Spark broadcast variables.
		If there is an error when process the row, then the return is a list with '' as the only
		item and an error is printed.
		"""
		body_list = []
		try:
			for word in row.lower().split(): 
				word = rgx.value.sub('', word) 
				if word not in stop_words_broadcast.value: 
					body_list.append(stemmer.value.stem(word))
			return body_list
		except Exception as e:
			print(e)
			return ['']

	# Declaring broadcast variables/functions to optimize performance.
	stemmer = sc.broadcast(PorterStemmer())
	strip_chars = ".?,!;:\"'()" 
	rgx = sc.broadcast(re.compile('[%s]' % strip_chars))

	# Creates a udf
	process = udf(process_str, ArrayType(StringType()))
		
	# Creation of Spark Context and Session
	sc = SparkContext()
	spark = SparkSession.builder.appName("SimpleApp").getOrCreate()
	print('*'*60, '\n', sc.getConf().getAll(), '\n', '*'*60, '\n')

	# Because dataframe is so large it must be repartitioned to utilize all nodes in the network.
	# 
	repartition_num = 1000

	# This s3 path contains all of the amazon reviews in all categories. Spark will
	# automatically iterate through all files in the 'tsv' folder. It then
	data = sc.textFile('s3://amazon-reviews-pds/tsv/')
	df = spark.read.csv(data, sep="\t", header=True, inferSchema=True)
	df = df.repartition(repartition_num)

	# removes punctuation, adds column 'body_list' which is a list of words, and selects only the star_rating and body_list columns
	stop_words = set(stopwords.words('english'))
	for word in ['', 1, 2, 3, 4, 5]:
		stop_words.add(word)



	df_new = df.withColumn('body_list', process(df['review_body']))\
			.select('star_rating', 'body_list')

	print("-"*30, '\nFinished creating new_df\n', "-"*30)
	print(df_new.count())

	cv = CountVectorizer(inputCol="body_list", outputCol="features")
	print("created cv line 67")
	idf = IDF(inputCol='features', outputCol = 'idf_features')
	print("created idf line 69")
	#rf = RandomForestClassifier(labelCol='star_rating', featuresCol='idf_features', seed=100)
	print("created rf line 71")

	print('started countvectorizer on line 73')
	# Stage one in pipeline the words are separated into count vectors
	stage1_fit = cv.fit(df_new)
	stage1_transform = stage1_fit.transform(df_new)
	stage1_transform.repartition(repartition_num)
	print('*'*40 + "Head of file:")
	print(df.head())
	print('*'*40 + "Head of processed:")
	print(stage1_transform.take(5))
	print('*'*40+"end of script")

	#Stage two in pipeline for tf-IDF
	stage2_fit = idf.fit(stage1_transform)
	stage2_transform = stage2_fit.transform(stage1_transform)
	
	print('*'*40+'\n', stage2_transform.select('star_rating', 'idf_features').take(1), '\n'+'*'*40)
	print('*'*40+'\n', stage2_transform.select('star_rating', 'idf_features').count(), '\n'+'*'*40)
	print('*'*14 + 'END OF SCRIPT'+'*'*14)

	# Stage three in pipeline for random forest
	'''stage3_fit = rf.fit(stage2_transform)
	stage3_transform = stage3_fit.transform(stage2_transform)
	print("-"*30, '\nFinished stage 3\n', "-"*30)

	def print_top_features(model, n=10):
	"""
	Input: RandomForest model, number of top feature words to print.
	Output: None. Prints top n words to console.
	"""
	sorted_indices = np.argsort(model.featureImportances.toArray())[-1:-(n+1):-1]
	for i in range(n):
			print(stage1_fit.vocabulary[sorted_indices[i]])
	print_top_features(stage3_fit)'''

	"""pd = stage3_transform.select('star_rating', 'idf_features').toPandas()
	pd.to_pickle('idf_pandas')
	print("-"*30, '\nsaved pandas to a pickle file\n', "-"*30)"""

if __name__ == "__main__":
    main()