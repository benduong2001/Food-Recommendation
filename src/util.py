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



import findspark
import pyspark
import pyspark.sql.functions as F
import pyspark.sql.types as T
import pyspark.ml as M
from pyspark.sql.window import Window
from pyspark.sql import SparkSession

def get_negative_words_of_stop_words(stop_words):
    negative_words = ["no","not","t","nor","against"] # constant
    contraction_not_words = [word for word in stop_words if word[-2:]=="'t"]
    contraction_not_words_pre_apostrophe = [word[:-2] for word in contraction_not_words]
    negative_words += contraction_not_words
    negative_words += contraction_not_words_pre_apostrophe
    return negative_words

def temp_interaction_metrics(temp_df, column_name_a, column_name_b):
    print("{0}s by {1}".format(column_name_b, column_name_a))
    b_by_a = temp_df.groupby([column_name_a],as_index=False).agg({column_name_b: set})
    print("{0} amount :".format(column_name_a), b_by_a.shape[0])
    b_by_a_len = b_by_a[column_name_b].apply(len)
    print("max {0} amount of a {1} :".format(column_name_b, column_name_a), b_by_a_len.max())
    print("median {0} amount of a {1} :".format(column_name_b, column_name_a), b_by_a_len.median())
    print("mean {0} amount of a {1} :".format(column_name_b, column_name_a), b_by_a_len.mean())
    fig, axs = plt.subplots()
    axs.hist(b_by_a_len)
    return b_by_a
#user_ids_by_business_ids = temp_interaction_metrics(reviews_df, "business_id", "user_id")

def temp_z_score_accom(x):
    temp_mean = np.mean(x)
    temp_std = np.std(x, ddof=0)
    if temp_std == 0:
        result = 0
    else:
        result = (x - temp_mean)/(temp_std)
    return result
def get_relative_groupwise_numbers(temp_df, group_column_name, numbers_column_name):
    result = temp_df.groupby([group_column_name])[numbers_column_name].transform(temp_z_score_accom)
    return result
def get_subjective_ratings(temp_df, user_id_column_name, rating_column_name):
    #rating_column = temp_df[rating_column_name]
    temp_means = temp_df.groupby([user_id_column_name])[rating_column_name].transform(np.mean)
    temp_stdevs = temp_df.groupby([user_id_column_name])[rating_column_name].transform(lambda x: np.std(x, ddof=0))
    #temp_stdev_numerator_presum = (rating_column - temp_means)**2
    #temp_stdev_numerator_sum = temp_df_.groupby([user_id_column_name])["temp_stdev_numerator_presum"].transform(sum)
    #temp_stdev_denominator = temp_df.groupby([user_id_column_name])[rating_column_name].transform(len)
    #temp_variance = 
    temp_z_score = temp_means / temp_stdevs
    temp_z_score_zerodiv = np.nan_to_num(temp_z_score, nan=0, posinf=0, neginf=0)
    result = temp_z_score_zerodiv
    return result
def get_normalized_ratings(temp_df, user_id_column_name, rating_column_name):
    rating_column = temp_df[rating_column_name]
    temp_means = temp_df.groupby([user_id_column_name])[rating_column_name].transform(np.mean)
    #temp_stdevs = temp_df.groupby([user_id_column_name])[rating_column_name].transform(lambda x: np.std(x, ddof=0))
    temp_stdev_numerator_presum = (rating_column - temp_means)**2
    temp_df_ = pd.DataFrame()
    temp_df_[user_id_column_name] = temp_df[user_id_column_name]
    temp_df_["temp_stdev_numerator_presum"] = temp_stdev_numerator_presum
    temp_stdev_numerator_sum = temp_df_.groupby([user_id_column_name])["temp_stdev_numerator_presum"].transform(sum)
    temp_stdev_denominator = temp_df.groupby([user_id_column_name])[rating_column_name].transform(len)
    temp_variance = temp_stdev_numerator_sum/temp_stdev_denominator
    temp_stdevs = np.sqrt(temp_variance)
    temp_z_score = temp_means / temp_stdevs
    temp_z_score_zerodiv = np.nan_to_num(temp_z_score, nan=0, posinf=0, neginf=0)
    result = temp_z_score_zerodiv
    return result

def word2vec_arithmetic_data(word2vec_model, group1, group1_version_word, group2):
    #group1 = "india" # man
    #group1_version_word = "naan bread" # king
    #group2 = "arabic" # woman
    result = word2vec_model.wv.most_similar(positive=[group2, group1_version_word], negative=[group1])
    return pd.Series(index=[x[0] for x in result], data=[x[1] for x in result])[::-1]
def word2vec_arithmetic(word2vec_model, group1, group1_version_word, group2):
    #group1 = "india" # man
    #group1_version_word = "naan bread" # king
    #group2 = "arabic" # woman
    title = ("{0} - {1} + {2} = ".format(group1_version_word, group1, group2))
    result = word2vec_arithmetic_data(word2vec_model, group1, group1_version_word, group2)
    return result.plot(kind="barh", title=title)

def dataclean_strlistnum_to_listfloat(
    column: pd.Series
                                   ) -> pd.Series:
    """
    If the column has values that are strings of list of numbers, rather than list of numbers,
    this turns it into a list of floats. "[1, 2.0, 3.2]" becames [1.0, 2.0, 3.2]. 
    Args:
        column (pd.Series): column is pd.Series where each value is a string of a list of numbers
    Returns:
        pd.Series where each value is a list of floats
    """    
    result = (
    column
    ).str.strip("]["
    ).str.replace(",",""
    ).str.split(" "
    ).apply(lambda l: [float(x) for x in l])
    return result

def dataclean_explodes_list_to_cols(
    column: pd.Series
) -> pd.DataFrame:
    """
    If the column has lists as values, this outputs a dataframe that explodes the columns.
    EACH LIST IN THE COLUMN NEEDS TO HAVE THE SAME SIZE
    Args:
        column (pd.Series): column is pd.Series where each value is a list of the same size throughout
    Returns:
        pd.DataFrame
    """
    column_name = column.name
    amount = len(column.values[0])
    result = pd.DataFrame(np.array(column.tolist()),
                          columns=[str(column_name)+"_"+str(i) for i in range(amount)]
                         )
    return result

def dataclean_strliststr_to_liststr(
    column: pd.Series
                                   ) -> pd.Series:
    """
    If the column has values that are strings of list of strings, rather than list of strings,
    this turns it into a list of strings. "['a','b','c']" becames ['a','b','c']. 
    Args:
        column (pd.Series): column is pd.Series where each value is a string of a list of strings
    Returns:
        pd.Series where each value is a list of strings
    """
    result = (
      column
      ).str.strip("]["
      ).str.replace("""'""",""
      ).str.split(", "
    )
    return result

def dataclean_everygram_str_to_liststr(
    column: pd.Series,
    max_len = 1,
                                      ) -> pd.Series:
    """
    Applies everygrams onto the column of the raw_recipes csv. Uses whitespace to split.
    Example:
    if max_len = 2,then "grilled cheese sandwich" -> ["grilled", "grilled cheese", "cheese", "cheese sandwich", "sandwich"]
    Args:
        column (pd.Series): column is pd.Series where each value is raw text. 
    Returns:
        pd.Series where each value is a list of strings
    """    
    assert max_len >= 1
    column = (column).apply(lambda x: re.sub(" +", " ",str(x))).str.split(" ")
    result = column.apply(lambda s: [" ".join(list(t)) for t in nltk.everygrams(s, max_len=2)])
    return result

def dataclean_join_liststr_columns(
    list_of_columns: list
) -> pd.Series:
    """
    list_of_columns is List<pd.Series<List<String>>>
    This Horizontally concatenates a list of columns where the values are lists.
    Args:
        list_of_columns (list): list of columns
    Returns:
        pd.Series - a column where each value is a giant list, which is the row-wise concatenations of the list columns.
    """
    result = list_of_columns[0]
    for column in list_of_columns[1:]:
        result += (column)
    return result
def dataclean_joined_liststr_columns_to_str(
    liststr_column: pd.Series
) -> pd.Series:
    """
    liststr_column is pd.Series<List<String>>. This joins the strings
    Args:
        liststr_column (pd.Series): column of list of strings
    Returns:
        pd.Series - a column where each value is a string, which is the row-wise concatenations of the liststr columns.
    """
    result = (liststr_column).apply(lambda x: " ".join(x))
    return result
def dataclean_word2vec(
    column,
    size =  100,
    window = 3,
    min_count = 1,
                      ):
    """
    creates the word2vec model
    Args:
        column (pd.Series): column is pd.Series where each value is a list of strings. Word2Vec requires column to be this format
    Returns:
        gensim.models.Word2Vec
    """
    word2vec_model = gensim.models.Word2Vec(
            column,
            size=size,
            window=window,
            min_count=min_count,
            workers=10
    )
    return word2vec_model

def clean_text_column(text_column):
    non_alphanumeric = string.punctuation # constant
    text_column = text_column.str.lower()
    text_column = text_column.str.translate(str.maketrans(non_alphanumeric, " "*len(non_alphanumeric)))
    return text_column






# vectorizer method, stopwords, max_df
# liststr_columns

class Temp_Recipe_Vectorizer_Builder:
    def __init__(self,
                 vectorizer=None,
                 food_vocab_column=None
                 
                 ):
        self.vectorizer = None
        self.vectorized = None
        self.set_vectorizer(vectorizer)
        if (food_vocab_column is None) == False:
            self.set_up(vectorizer, food_vocab_column)
    def set_vectorizer(self, vectorizer=None):
        if vectorizer is None:
            vectorizer = Pipeline([('count', CountVectorizer(stop_words=stop_words)),
                                   ('tfidf', TfidfTransformer())])
        self.vectorizer = vectorizer
    def fit(self, food_vocab_column):
        fitted_vectorizer = self.vectorizer.fit(food_vocab_column)
        self.vectorizer = fitted_vectorizer
        return fitted_vectorizer
    def transform(self, food_vocab_column):
        vectorized = self.vectorizer.transform(food_vocab_column)
        self.vectorized = vectorized
        return vectorized
    def set_vocab2idx(self, vocab):
        if vocab is None: 
            vocab = self.vocab
        vocab2idx = vocab
        self.vocab2idx = vocab2idx
        return vocab2idx
    def set_idx2vocab(self, vocab=None):
        if vocab is None: 
            vocab = self.vocab
        idx2vocab = dict((v, k) for k, v in vocab.items())
        self.idx2vocab = idx2vocab
        return idx2vocab
    def set_vocab(self, vocab=None):
        if vocab is None: 
            vocab = self.vocab
        vocab2idx = self.set_vocab2idx(vocab)
        idx2vocab = self.set_idx2vocab(vocab)
        self.vocab2idx = vocab2idx
        self.idx2vocab = idx2vocab
    def set_up(self, vectorizer=None, food_vocab_column=None):
        if vectorizer is None:
            vectorizer = self.vectorizer
        self.set_vectorizer(vectorizer)
        self.fit(food_vocab_column)
        self.transform(food_vocab_column)
        
        vocab_holder = self.vectorizer
        if type(vocab_holder) == Pipeline: 
            vocab = vocab_holder.named_steps["count"].vocabulary_
        else: 
            vocab = vocab_holder.vocabulary_
        self.vocab = vocab
        self.set_vocab(vocab)



def get_liststr_columns(raw_recipes_df, column_names=["ingredients","name"]):
    liststr_columns_list = []
    if "tags" in column_names:
        cleaned_tag_column = dataclean_strliststr_to_liststr(raw_recipes_df["tags"])
        liststr_columns_list.append(cleaned_tag_column)
    if "ingredients" in column_names:
        cleaned_ingredients_column = dataclean_strliststr_to_liststr(raw_recipes_df["ingredients"])
        liststr_columns_list.append(cleaned_ingredients_column)
    if "name" in column_names:
        cleaned_name_column = dataclean_everygram_str_to_liststr(raw_recipes_df["name"],2)
        liststr_columns_list.append(cleaned_name_column)
    if "steps" in column_names:
        cleaned_steps_column = dataclean_strliststr_to_liststr(raw_recipes_df["name"])
        liststr_columns_list.append(cleaned_steps_column)
    if "description" in column_names:
        cleaned_description_column = dataclean_everygram_str_to_liststr(raw_recipes_df["name"],2)
        liststr_columns_list.append(cleaned_description_column)
    liststr_columns = dataclean_join_liststr_columns(liststr_columns_list)
    return liststr_columns



class Tfidf_Group_Collapser:
    def __init__(
        self, 
        temp_reviews_df,
        label_column_name,
        review_text_column_name,
        vectorizer,
        temp_labels=None
    ):            
        self.temp_reviews_df = temp_reviews_df
        self.label_column_name = label_column_name
        self.review_text_column_name = review_text_column_name
        self.vectorizer = vectorizer 
        # vectorizer should be a temp_recipe_vectorizer_builder_object
        # has pipeline vocabulary = recipe_tfidf, stop words = english, but NOT fitted NOR transformed yet
        if temp_labels is None:
            self.temp_labels = list(pd.unique(self.temp_reviews_df[self.label_column_name]))
        self.review_text_column = clean_text_column(self.temp_reviews_df[self.review_text_column_name])
    
    def fit_transform_vectorizer(self, review_text_column=None):
        # review_text_column should already be cleaned!
        if review_text_column is None:
            review_text_column = self.review_text_column
        else:
            review_text_column = clean_text_column(review_text_column)
        #self.vectorizer.fit(review_text_column)
        #vectorized = self.vectorizer.transform(review_text_column)
        self.vectorizer.set_up(vectorizer=None, food_vocab_column=review_text_column)
        vectorized = self.vectorizer.vectorized
        self.vectorized = vectorized
        return vectorized
        
    def collapse_by_mean(self, temp_labels=None):
        self.fit_transform_vectorizer()
        
        if temp_labels is None: 
            temp_labels = self.temp_labels
        
        self.temp_labels = temp_labels
        vectorized = self.vectorized
        temp_reviews_df = self.temp_reviews_df
        label_column_name = self.label_column_name
        
        vectorized_collapsed = []
        
        for temp_label in temp_labels:
            temp_label_vectorized_collapsed = (
                vectorized[np.where(temp_reviews_df[label_column_name]==temp_label)[0]].mean(axis=0)
            )
            vectorized_collapsed.append(np.squeeze(temp_label_vectorized_collapsed.A,0))
        vectorized_collapsed = np.array(vectorized_collapsed)
        self.vectorized_collapsed = vectorized_collapsed
        return vectorized_collapsed
    
    def collapse_by_strjoin(self, temp_labels=None): 
        if temp_labels is None: 
            temp_labels = self.temp_labels
        self.temp_labels = temp_labels
        vectorizer = self.vectorizer
        temp_reviews_df = self.temp_reviews_df
        label_column_name = self.label_column_name
        review_text_column_name = self.review_text_column_name
        
        grouped_review_text_column = []

        for label in temp_labels:
            label_all_text = ""
            temp_reviews_df_label = temp_reviews_df[temp_reviews_df[label_column_name]==label]
            temp_review_text_column = temp_reviews_df_label[review_text_column_name]
            temp_review_text_column = clean_text_column(temp_review_text_column)
            for row in tqdm.tqdm(list(temp_review_text_column.values)):
                label_all_text += (" "+row)
            grouped_review_text_column.append(label_all_text)
        grouped_review_text_column = pd.Series(grouped_review_text_column)
        assert len(grouped_review_text_column) == len(temp_labels)
        vectorized_collapsed = self.fit_transform_vectorizer(grouped_review_text_column)
        assert (vectorized_collapsed.shape[0]) == len(temp_labels)
        self.vectorized_collapsed = vectorized_collapsed
        return vectorized_collapsed
    def build_temp_word2label_df(self):
        temp_word2label_df = pd.DataFrame(self.vectorized_collapsed.A.T, columns=self.temp_labels)
        temp_word2label_df["word"] = [self.vectorizer.idx2vocab[i] for i in range(self.vectorized_collapsed.shape[1])]
        temp_word2label_df["arg_label"] = np.argmax(temp_word2label_df[self.temp_labels].values,axis=1)
        temp_word2label_df["stdev"] = np.std(temp_word2label_df[self.temp_labels].values,axis=1,ddof=0)
        return temp_word2label_df
    
#word 1st, label 2nd       
#sort wordidx by most stdev
#









class Reviews_Dataset_Reader:
    def __init__(self, 
                 configs=None,
                 spark=None,
                ):
        if configs is None:
            path_folder = "../../data/"
            dataset_version = "FILTER",
            using_pics = 0
            using_spark = 0
            pic_link_header = "https://lh5.googleusercontent.com/p/"
        else:
            path_folder = configs["Reviews_Dataset_Reader"]["path_folder"]
            dataset_version = configs["Reviews_Dataset_Reader"]["dataset_version"]
            using_pics = configs["Reviews_Dataset_Reader"]["using_pics"]
            using_spark = configs["Reviews_Dataset_Reader"]["using_spark"]
            pic_link_header = configs["Reviews_Dataset_Reader"]["pic_link_header"]
        assert dataset_version in ["FULL", "FILTER"]
        assert using_pics in [0,1]    
        #self.path_folder = path_folder
        self.dataset_version = dataset_version
        self.using_pics = using_pics
        if self.using_pics == 1:
            self.dataset_version = "FILTER"
            self.vggnet_model = VGG16()
            self.pic_link_header = pic_link_header
        self.using_spark = using_spark
        if self.using_spark == 1:
            self.spark = spark
        if self.dataset_version == "FULL":
            file_name_data = "image_review_all.json"
        if self.dataset_version == "FILTER":
            file_name_data = "filter_all_t.json"
        self.path_folder_data = os.path.join(path_folder,"raw",file_name_data)

    def get_pics_words_from_links(self, pic_links, top_n=10):
        pics_len = len(pic_links)
        vggnet_model_input = []
        for i in (range(0, pics_len)):
            try:
                temp_pic_link = pic_links[i]
                temp_pic_link = self.pic_link_header + temp_pic_link
                response = requests.get(temp_pic_link)
                image = Image.open(BytesIO(response.content))
                image = np.array(image)
                image =  np.array(Image.fromarray(image).resize((224, 224), Image.NEAREST))
                assert image.shape == (224, 224, 3)
                vggnet_model_input.append(image)
            except:
                pass
        vggnet_model_input = np.array(vggnet_model_input)
        if vggnet_model_input.size == 0:
            return []
        #print(vggnet_model_input.shape)
        assert list(vggnet_model_input.shape)[-3:] == [224, 224, 3]
        assert len(vggnet_model_input.shape) == 4
        #assert len(unbroken_indices) == vggnet_model_input.shape[0]
        encoded_predictions = self.vggnet_model.predict(vggnet_model_input) #, verbose=0)
        decoded_output = decode_predictions(encoded_predictions)
        
        pics_words = []
        for decoded_i in range(len(decoded_output)):
            pic_words = [label[1] for label in decoded_output[decoded_i][:top_n]]
            pics_words += pic_words
            # assert len(pics_words) > 0
        return pics_words
    def get_review_info(self, review):
        temp_business_id = None
        temp_user_id = None
        temp_rating = None
        temp_review_text = None
        temp_pics = None

        if "business_id" in review:
            temp_business_id = review["business_id"]
        if "user_id" in review:
            temp_user_id = review["user_id"]
        if "rating" in review:
            temp_rating = review["rating"]
        if "review_text" in review:
            temp_review_text = review["review_text"]
        if "pics" in review:
            temp_pics = review["pics"]
            if self.dataset_version == "FULL":
                temp_pics = [x["id"] for x in (temp_pics)]

        if self.using_pics == 1:
            if (temp_pics is None)==False:
                temp_pics_words = self.get_pics_words_from_links(temp_pics)
                if temp_review_text is None:
                    temp_review_text = ""
                temp_review_text += (" " + " ".join(temp_pics_words))
        return (temp_business_id, temp_user_id, temp_rating, temp_review_text)
    def build_reviews_df(self):
        
        raw_data_orig = []
        with open(self.path_folder_data) as f:
            for line in tqdm.tqdm(f):
                raw_data_orig.append(eval(line))
            f.close()
            
        assert len(raw_data_orig) == 1
        #print(raw_data_orig[0].keys())

        if self.dataset_version in ["FILTER"]:
            raw_data_train = (raw_data_orig[0]["train"])
            raw_data_val = (raw_data_orig[0]["val"])
            raw_data_test = (raw_data_orig[0]["test"])
            raw_data_sources = [raw_data_train, raw_data_val, raw_data_test]
        else:
            raw_data = raw_data
            raw_data_sources = [raw_data]
        
        reviews_df_business_id_column_list = []
        reviews_df_user_id_column_list = []
        reviews_df_rating_column_list = []
        reviews_df_review_text_column_list = []
        
        for raw_data_source in raw_data_sources:
            for review in tqdm.tqdm(raw_data_source):
                temp_business_id, temp_user_id, temp_rating, temp_review_text = self.get_review_info(review)

                reviews_df_business_id_column_list.append(temp_business_id)
                reviews_df_user_id_column_list.append(temp_user_id)
                reviews_df_rating_column_list.append(temp_rating)
                reviews_df_review_text_column_list.append(temp_review_text)            

        reviews_df = pd.DataFrame()
        reviews_df["business_id"] = reviews_df_business_id_column_list
        reviews_df["user_id"] = reviews_df_user_id_column_list
        reviews_df["rating"] = reviews_df_rating_column_list
        reviews_df["review_text"] = reviews_df_review_text_column_list  
        
        business_id_label_encoder = LabelEncoder()
        user_id_label_encoder = LabelEncoder()

        business_id_label_encoder.fit(reviews_df["business_id"])
        user_id_label_encoder.fit(reviews_df["user_id"])

        reviews_df["business_id"] = business_id_label_encoder.transform(reviews_df["business_id"])
        reviews_df["user_id"] = user_id_label_encoder.transform(reviews_df["user_id"])
        self.reviews_df = reviews_df
        return reviews_df
    def convert_df_to_spark(self, reviews_df=None):
        if word2vec_df is None:
            reviews_df = self.reviews_df
        
        if self.using_spark == 1:
            reviews_df["review_text"].fillna("_",inplace=True)
            schema = T.StructType([
                T.StructField("business_id", T.IntegerType(), True),
                T.StructField("user_id", T.IntegerType(), True),
                T.StructField("rating", T.IntegerType(), True),
                T.StructField("review_text", T.StringType(), True)]
            )
            reviews_df = self.spark.createDataFrame(reviews_df, schema)        
        else:
            raise Exception("Spark is disabled; cannot convert dataframe to spark")
        
        self.reviews_df = reviews_df
        
        return reviews_df



class Word2vec_Dataset_Reader:
    def __init__(self, 
                 word2vec_model, 
                 lexicon,
                 configs = None,
                 spark = None,
                ):
        if configs is None:
            vector_mode = "COMPRESSED"
            using_spark = 0
        else:
            vector_mode = configs["Word2vec_Dataset_Reader"]["vector_mode"]
            using_spark = configs["Word2vec_Dataset_Reader"]["using_spark"]
        if using_spark == 1:
            vector_mode = "SPLIT"
        assert vector_mode in ["COMPRESSED","SPLIT"]
        self.word2vec_model = word2vec_model
        self.lexicon = lexicon
        self._vector_mode = vector_mode
        embedding_size = self.word2vec_model.vector_size
        if self._vector_mode == "COMPRESSED":
            self.vec_column_names = ["vec"]
        elif self._vector_mode == "SPLIT":
            self.vec_column_names = ["v{0}".format(i) for i in range(embedding_size)]
        self.using_spark = using_spark

        if self.using_spark == 1:
            self.spark = spark
    def build_word2vec_df(self):
        word2vec_vocab = self.word2vec_model.wv.vocab
        vocab = [word for word in word2vec_vocab if (word in self.lexicon)]
        vecs = [self.word2vec_model.wv[word] for word in vocab]
        vecs_arr = np.array(vecs)
        
        word2vec_df = pd.DataFrame()
        word2vec_df["review_text"] = vocab
        vec_column_names = self.vec_column_names
        vector_mode = self._vector_mode
        if vector_mode == "COMPRESSED":
            word2vec_df[vec_column_names[0]] = vecs
        elif vector_mode == "SPLIT":
            for i in range(len(vec_column_names)):
                vec_column_name = vec_column_names[i]
                word2vec_df[vec_column_name] = vecs_arr[:, i]
        self.word2vec_df = word2vec_df
        return word2vec_df
    def convert_df_to_spark(self, word2vec_df=None):
        if word2vec_df is None:
            word2vec_df = self.word2vec_df
        vector_mode = self.vector_mode
        vec_column_names = self.vec_column_names
        if self.using_spark == 1:
            if vector_mode == "COMPRESSED":
                schema = T.StructType([
                    T.StructField("review_text", T.StringType(), True)
                ]+[T.StructField(vec_column_names[0],T.ArrayType(T.DoubleType()), True)]
                )
            elif vector_mode == "SPLIT":
                schema = T.StructType([
                    T.StructField("review_text", T.StringType(), True)
                ]+[T.StructField(c, T.DoubleType(), True) for c in vec_column_names]
                )
            word2vec_df = self.spark.createDataFrame(word2vec_df, schema)
        else:
            raise Exception("Spark is disabled; cannot convert dataframe to spark")
        self.word2vec_df = word2vec_df
        return word2vec_df
