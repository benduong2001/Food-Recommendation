# -*- coding: utf-8 -*-
"""util.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lnieii2o5KKsixekF6KAOJin3Aubldsp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

import os
from bs4 import BeautifulSoup  
import requests
import zipfile
import io

#import geopandas as gpd
#import shapely
import gensim
import re
import tqdm
import tensorflow as tf
import keras
import string

import re
import pickle
import os
from collections import defaultdict
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = stopwords.words("english")



import tqdm
import pickle

from keras.applications.vgg16 import VGG16
#vggnet_model = VGG16()
pics_link_header = "https://lh5.googleusercontent.com/p/"
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

from PIL import Image
import requests
from io import BytesIO

from sklearn.preprocessing import LabelEncoder




import src.util as util

import src.data.etl as etl
import src.data.build_tfidf as build_tfidf
import src.data.build_word2vec as build_word2vec
import src.features.data_preparation_interaction as dpi
import src.models.model_interaction as mi



import findspark
import pyspark
import pyspark.sql.functions as F
import pyspark.sql.types as T
import pyspark.ml as M
from pyspark.sql.window import Window
from pyspark.sql import SparkSession

SPARK_SESSION_VERSION = 1

def spark_builder():
    findspark.init()
    if SPARK_SESSION_VERSION == 1:
        spark = SparkSession.builder.getOrCreate()
    elif SPARK_SESSION_VERSION == 2:
        spark = SparkSession.builder \
            .master('local[*]') \
            .config("spark.driver.memory", "3g") \
            .appName('food_rec') \
            .getOrCreate()
    return spark
def main_data(configs):
    etl.main()
    recipe_tfidf = build_tfidf.main(configs=configs, method="load")
    recipe_word2vec = build_word2vec.main(configs=configs, method="load")
    return recipe_tfidf, recipe_word2vec
def main(configs):
    spark = None
    recipe_tfidf, recipe_word2vec = main_data(configs)
    if configs["Reviews_Dataset_Reader"]["using_spark"] == 1:
        spark = spark_builder()
    reviews_dataset_reader = util.Reviews_Dataset_Reader(path_folder, configs=configs, spark=spark)
    reviews_df = reviews_dataset_reader.build_reviews_df()

    stop_words = stopwords.words('english')
    lexicon = [word for word in recipe_tfidf.vocab2idx if word not in stop_words]
    word2vec_dataset_reader = util.Word2vec_Dataset_Reader(
        recipe_word2vec,
        lexicon,
        configs=configs,
        spark = spark
    )
    word2vec_df = word2vec_dataset_reader.build_word2vec_df()
    
    if configs["Temp_Interaction_Data_Preparation_Builder"]["using_spark"] == 0:
        temp_interaction_data_preparation_builder = dpi.Temp_Interaction_Data_Preparation_Builder(
            recipe_tfidf,
            word2vec_df,
            configs=configs,
            spark=spark
        )
        temp_interaction_model_builder = mi.Temp_Interaction_Model_Builder(
            temp_interaction_data_preparation_builder
        )
    else:
        temp_interaction_data_preparation_builder = dpi.Temp_Interaction_Data_Preparation_Builder_Pyspark(
            recipe_tfidf,
            word2vec_df,
            configs=configs,
            spark=spark
        )        
        temp_interaction_model_builder = mi.Temp_Interaction_Model_Builder_Pyspark(
            temp_interaction_data_preparation_builder
        )
    temp_interaction_model_builder.baseline()

if __name__ == "__main__":
    with open("configs.json", "rb") as f:
        configs = json.load(f)
    main(configs)








    



