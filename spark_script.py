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

start_time = time.time()

progress = open('progress.txt', 'w')

sc = SparkContext()
spark = SparkSession.builder.appName("SimpleApp").getOrCreate()

data = sc.textFile('s3://amazon-reviews-pds/tsv/amazon_reviews_us_Video_Games_v1_00.tsv.gz')
df = spark.read.csv(data, sep="\t", header=True, inferSchema=True)
progress.write('---Loaded data into dataframe---\n')

stop_words = set(stopwords.words('english'))
stop_words.add('')

# removes punctuation, adds column 'body_list' which is a list of words, and selects only the star_rating and body_list columns
stop_words = set(stopwords.words('english'))
for word in ['', 1, 2, 3, 4, 5]:
    stop_words.add(word)

def process_str(row):
    """Input: String
    Output: List of strings without punctuation
    
    Removes punctuation using functions from the strings library and stems using NLK. 
    ."""
    try:
        stemmer = PorterStemmer()
        strip_chars = ".?,!;:\"'()" 
        rgx = re.compile('[%s]' % strip_chars) 

        body_list = []
        for word in row.lower().split(): 
            word = rgx.sub('', word) 
            if word not in stop_words: 
                body_list.append(stemmer.stem(word))
        
        return body_list
    except Exception as e:
        progress.write(e)
        return []

process = udf(process_str, ArrayType(StringType()))
print("line 64")
df_new = df.withColumn('body_list', process(df['review_body']))\
        .select('star_rating', 'body_list').limit(5000)
print("line 67")
cv = CountVectorizer(inputCol="body_list", outputCol="features", minDF=0.001)
idf = IDF(inputCol='features', outputCol = 'idf_features')
rf = RandomForestClassifier(labelCol='star_rating', featuresCol='idf_features', seed=100)

# Stage one in pipeline the words are separated into count vectors
stage1_fit = cv.fit(df_new)
stage1_transform = stage1_fit.transform(df_new)
progress.write("---Finished Stage 1---\n")

# Stage two in pipeline for tf-IDF
stage2_fit = idf.fit(stage1_transform)
stage2_transform = stage2_fit.transform(stage1_transform)
progress.write("---Finished Stage 2---\n")

# Stage three in pipeline for random forest
stage3_fit = rf.fit(stage2_transform)
stage3_transform = stage3_fit.transform(stage2_transform)
progress.write("---Finished Stage 3---\n")

def print_top_features(model, n=10):
    """
    Input: RandomForest model, number of top feature words to print.
    Output: None. Prints top n words to console.
    """
    sorted_indices = np.flip(np.argsort(model.featureImportances.toArray()))
    for i in range(20):
        print(stage1_fit.vocabulary[sorted_indices[i]])
print_top_features(stage3_fit)

pd = stage3_transform.select('star_rating', 'idf_features').toPandas()
pd.to_pickle('idf_pandas')

time_spent = time.time() - start_time
progress.write('File took {:.2f} seconds to run'.format(time_spent))
progress.close()