#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging as logger

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import logging
from sklearn.metrics import roc_auc_score

from sklearn.metrics import confusion_matrix
import seaborn as sns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

import os
from collections import defaultdict
from bs4 import BeautifulSoup  
import requests
import zipfile
import io
import nltk
from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
#from nltk.tokenize import word_tokenize
nltk.download('stopwords')
stop_words = stopwords.words('english')
#import geopandas as gpd|
#import shapely
import gensim
import tqdm
import tensorflow as tf
import keras

from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import normalize
import sys

path_folder = (os.path.abspath(os.path.join((os.path.abspath(os.path.join(os.getcwd(), os.pardir))),os.pardir)))

google_colab = 0
if google_colab == 1:
    from google.colab import drive
    drive.mount('/content/drive/')
    path_folder = "/content/drive/MyDrive/dsprojects/dsproject_grev/"
    
sys.path.insert(0, path_folder+"/src/"#+features/"
                )
import util

import tqdm
import pickle

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from keras.applications.vgg16 import VGG16
#vggnet_model = VGG16()
pics_link_header = "https://lh5.googleusercontent.com/p/"
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

from PIL import Image
import requests
from io import BytesIO

from sklearn.preprocessing import LabelEncoder

import string

import findspark
import pyspark
import pyspark.sql.functions as F
import pyspark.sql.types as T
import pyspark.ml as M
import pyspark.mllib as MLB

from pyspark.sql.window import Window
from pyspark.sql import SparkSession
    
class Temp_Interaction_Data_Preparation_Builder:
    def __init__(
        self,
        recipe_tfidf,
        word2vec_df,
        configs = None,
        spark = None
    ):
        if configs is None:
            with_tfidf_weighting = 1
            with_concatenated = 1
            using_spark = 0
        else:
            with_tfidf_weighting = configs["Temp_Interaction_Data_Preparation_Builder"]["with_tfidf_weighting"]
            with_concatenated = configs["Temp_Interaction_Data_Preparation_Builder"]["with_concatenated"]
            using_spark = configs["Temp_Interaction_Data_Preparation_Builder"]["using_spark"]
        self.recipe_tfidf = recipe_tfidf
        self.word2vec_df = word2vec_df
        self.with_tfidf_weighting = with_tfidf_weighting
        self.with_concatenated = with_concatenated
        self.using_spark = using_spark
        if self.using_spark == 1:
            self.spark = spark
        #self.id_df = pd.DataFrame(columns=["user_id","business_id","review_id"])
        #self.review_df = pd.DataFrame(columns=["review_id","review_text"])
        self.curr_review_id_count = 0
    def split_df_id_review(
        self, 
        reviews_df
    ):
        # ONLY RUN ONCE AT THE START
        # reviews_df contains the y column!
        # reviews_df must be all positives, and before train/test split
        
        new_review_id_count = self.curr_review_id_count + reviews_df.shape[0]

        reviews_df["review_id"] = np.arange(self.curr_review_id_count, new_review_id_count)
        id_df = reviews_df[["user_id","business_id","review_id"]]
        review_df = reviews_df[["review_id","review_text"]]
        self.id_df = id_df
        self.review_df = review_df
    def build_review2word(self, review_df=None):
        
        if review_df is None:
            review_df = self.review_df
        review_df["review_text"] = util.clean_text_column(review_df["review_text"]).str.split(" ")
        review2word_df = review_df.explode("review_text")
        logger.info("Exploded review_text")
        print("Exploded review_text")

        review2word_df = review2word_df[~(review2word_df["review_text"].isin(stop_words))]
        review2word_df = review2word_df[(review2word_df["review_text"].isin(self.word2vec_df["review_text"].values))]
        review2word_df = review2word_df[(review2word_df["review_text"].isin(self.recipe_tfidf.vocab2idx))]
        review2word_df = review2word_df[(review2word_df["review_text"] != "")]
        logger.info("Filtered exploded review_text")
        print("Filtered exploded review_text")

        self.review2word_df = review2word_df
        return review2word_df
    
    def temp_id_removal(self, 
                        #review2word_df=None
                       ):
        '''
        Removes words that aren't in the tf idf dictionary, or aren't in in the word2vec, or is in stop-words
        Afterwards, it removes any review_id that did not survive said operations
        It also removes review_ids where BOTH the user id and business id only appear 1 one time
        (since when we remove current samples from the averaging later on, it would pretty much remove the whole row)
        '''
        logger.info("temp_id_removal")
        print("temp_id_removal")

        review2word_df = self.review2word_df
        review_df = self.review_df
        id_df = self.id_df
        
        temp_valid_ids_after_exploded_review_text_filter = review2word_df[["review_id"]].drop_duplicates()
        review_df = review_df.merge(temp_valid_ids_after_exploded_review_text_filter, on=["review_id"])
        # filter out the rows with invalid review ids in old review_df,
        self.review_df = review_df
        
        # clean up id_df
        # filter out the rows with invalid review ids in old id_df,
        id_df = id_df.merge(temp_valid_ids_after_exploded_review_text_filter, on=["review_id"])
        # Remove double singletons
        one_time_business_id_list = id_df[["business_id"]].drop_duplicates(subset=["business_id"], keep=False)["business_id"].values
        one_time_user_id_list = id_df[["user_id"]].drop_duplicates(subset=["user_id"], keep=False)["user_id"].values
        id_df = id_df[
            ~( (id_df["business_id"].isin(one_time_business_id_list)) & (id_df["user_id"].isin(one_time_user_id_list)))
        ]
        self.id_df = id_df
        
        del one_time_business_id_list
        del one_time_user_id_list
        logger.info("Removed double singletons")
        print("Removed double singletons")

    
        # for the word-level, vec can join in now or after tfidf
        # for the word-level, tfidf needs to regroup the level and re-explode it.
        # this re-explosion stage must be separate and after the first explosion, due to the id-removal in between
        # (which also affects the groupby for the user or business ids in the re-explosion)
    
    def _get_tfidfs_sentence(self, tfidf_features_column, row_idx, words):
        return [(tfidf_features_column.iloc[row_idx][self.recipe_tfidf.vocab2idx[word]]) for word in words]
    
    def build_review2word2weight(self, temp_id_column_name, temp_id2review2word_df):
        '''
        Adds the weighting column (word_count) to the word-wise granularity level table
        This weighting column is how the weighted averages of the vector embeddings will be 
        In the usual circumstance ("unweighted"), the weight column is just the word count itself
        But when weighting is set to true, tf-idf weighting will be used
        '''
        # unless we have a word index column, we'd need the review_id to be exploded
        temp_id2review_df = temp_id2review2word_df.groupby([temp_id_column_name], as_index=False).agg({
            "review_id": list,
            "review_text": list})
        temp_id2review_df["review_text"] = temp_id2review_df["review_text"].apply(lambda x: " ".join(x))

        vectorizer = Pipeline([('count', CountVectorizer(
            vocabulary=self.recipe_tfidf.vocab2idx,
            stop_words=stop_words,
            max_df=1,
        )), ('tfidf', TfidfTransformer())])
        vectorizer.fit(temp_id2review_df['review_text'])
        tfidf_features = vectorizer.transform(temp_id2review_df["review_text"])##
        
        assert (temp_id2review_df.shape[0] == tfidf_features.shape[0])
        temp_id2review_df["review_text"] = temp_id2review_df["review_text"].str.split(" ")
        tfidf_features_column = pd.Series([tfidf_features[i].A[0] for i in range(tfidf_features.shape[0])])

        word_counts_column = [self._get_tfidfs_sentence(tfidf_features_column, i, words) 
                              for i, words 
                              in enumerate(temp_id2review_df["review_text"].values)]

        temp_id2review_df["word_count"] = word_counts_column

        temp_id2review2word2weight_df = temp_id2review_df.explode(["review_id", "review_text","word_count"])
        # now back at the same granularity, and same form as temp_id2review2word_df , but with a new weight column
        return temp_id2review2word2weight_df

    def build_review2word2vec(self, temp_id2review2word_df):
        '''
        Adds the word2vec column to the word-wise granularity level table
        by merging with the word2vec table (which maps words to their vector representations)
        '''
        temp_id2review2word2vec_df = temp_id2review2word_df.merge(self.word2vec_df, on=["review_text"])
        return temp_id2review2word2vec_df
    
    def build_word_level_df(self, temp_id_column_name, temp_id2review2word_df):
        '''
        This function will create the table for either user2review2word_df or  business2review2word_df
        Both tables are at the word-wise granularity level
        With columns user_id/business_id, review_id, review_text, word_count, and vector embedding
        Where [user_id/business_id, review_id, review_text] form a super key
        Thus each row is a word (and its tf-idf score, and its word2vec vector) of a review, of a user or business id
        '''
        logger.info(temp_id2review2word_df.columns)
        temp_id2review2word2weight_df = self.build_review2word2weight(temp_id_column_name, temp_id2review2word_df)  
        if self.with_tfidf_weighting == 0:
            # undo all tf-idf weighting (treat it as a regular unweighted average)
            temp_id2review2word2weight_df["word_count"] = 1
        temp_id2review2word2weight2vec_df = self.build_review2word2vec(temp_id2review2word2weight_df)
        temp_id2review2word_level_df = temp_id2review2word2weight2vec_df
        return temp_id2review2word_level_df
    
    def build_id2review2word_df(self):
        # ##### word-level is now [user_id, review_id, word, tfidf (vanilla as word_count)]
        review2word_df = self.review2word_df
        id_df = self.id_df
        
        #review2word_df.sort_values(["review_id"], inplace=True)
        id2review2word_df = id_df.merge(review2word_df, on=["review_id"])
        self.id2review2word_df = id2review2word_df
        return id2review2word_df
    def build_id2vec_df(self, id2review2word_df=None):
        '''
        Creates id2vec_df, which has columns ["user_id","business_id","review_id"]
        id2vec_df is an intermediate table from which the tables for 
        mapping user_id / business_id to vector embeddings will be created
        '''
        if id2review2word_df is None:
            id2review2word_df = self.id2review2word_df
        id2vec_df = id2review2word_df[["user_id","business_id","review_id"]].drop_duplicates()
        # In id2vec_df, review_id work as a unique primary key
        self.id2vec_df = id2vec_df
        return id2vec_df
    def build_word_level_dfs(self, id2review2word_df=None):
        '''
        This function will create the tables user2review2word_df,  business2review2word_df
        Both tables are at the word-wise granularity level
        With columns user_id/business_id, review_id, review_text, word_count, and vector embedding
        Where [user_id/business_id, review_id, review_text] form a super key
        Thus each row is a word (and its tf-idf score, and its word2vec vector) of a review, of a user or business id
        '''
        logger.info("build_word_level_dfs")
        print("build_word_level_dfs")

        if id2review2word_df is None:
            id2review2word_df = self.id2review2word_df
            
        id2review2word_df = id2review2word_df
        user2review2word_df = id2review2word_df[["user_id", "review_id","review_text"]]
        business2review2word_df = id2review2word_df[["business_id", "review_id","review_text"]]

        user2review2word2weight2vec_df = self.build_word_level_df("user_id", user2review2word_df)
        business2review2word2weight2vec_df = self.build_word_level_df("business_id", business2review2word_df)
        
        user2review2word_level_df = user2review2word2weight2vec_df
        business2review2word_level_df = business2review2word2weight2vec_df
        
        self.user2review2word_level_df = user2review2word_level_df
        self.business2review2word_level_df = business2review2word_level_df
        return user2review2word_level_df, business2review2word_level_df

    def aggregation_average_sum(
        self, 
        temp_id_version,
        temp_id2review2word_level_df,
        groupby_column_name,
    ):
        # temp_id_version is still usable here if splitting the vec to many columns
        temp_id_agg = (
            temp_id2review2word_level_df
            .groupby([groupby_column_name], as_index=False)
            .agg({"vec": list, "word_count": np.sum})
        )
        temp_id_agg["vec"] = temp_id_agg["vec"].apply(lambda x: np.sum(np.array(x), axis=0))
        return temp_id_agg
    def aggregation_average_sum_curr(
        self, 
        temp_id_version,
        temp_id2review2word_level_df
    ):  
        temp_id_agg = self.aggregation_average_sum(
            temp_id_version, 
            temp_id2review2word_level_df,
            temp_id_version+"_id", 
        )
        temp_id_agg = temp_id_agg.rename(
            columns={
                "vec": temp_id_version+"_vec", 
                "word_count": temp_id_version+"_word_count"}
        )
        return temp_id_agg
        
    def aggregation_average_divide(self, temp_id_version, agg_df):
        agg_df[temp_id_version + "_vec"] /= agg_df[temp_id_version + "_word_count"]
        return agg_df

    def aggregation_average_divide_curr(self, temp_id_version, agg_df):
        
        agg_df[temp_id_version+"_vec"] -= agg_df["vec"]
        agg_df[temp_id_version+"_word_count"] -= agg_df["word_count"]
        agg_df = agg_df[agg_df[temp_id_version+"_word_count"] != 0]
        agg_df = self.aggregation_average_divide(temp_id_version, agg_df)
        agg_df.drop(columns=["vec","word_count"],inplace=True)
        return agg_df
    
    def aggregation_averages(self):
        '''
        This function creates the weighted averages used for summarizing the user id's / business id's
        in terms of their words.
        Rather than directly column-wise averaging the word2vec vectors with an average function,
        this function breaks it apart as 2 separate operations (summing and dividing)
        
        2 reasons for why are:
        * Keeping the operations separated allows to subtract the current review aggregates from the overall review aggregates,
        which is necessary because otherwise the dataset for the prediction will accidentally give away the information. 
        For unobserved pairs, this is impossible to happen and so the regular averages 
        (without current mean subtraction) gets used.
        
        * To make thing extensible to new incoming data
        While simply doing 2 averages (with and without the current running rows) is an option, that feels too bulky
        '''
        logger.info("aggregation_averages")
        print("aggregation_averages")

        id2vec_df = self.id2vec_df
        user2review2word_level_df = self.user2review2word_level_df
        business2review2word_level_df = self.business2review2word_level_df

        logger.info("Aggregation for users, businesses")
        print("Aggregation for users, businesses")

        user_agg_sum_user = self.aggregation_average_sum_curr("user",user2review2word_level_df)
        user_agg_sum_review = self.aggregation_average_sum("user",user2review2word_level_df,"review_id")
        business_agg_sum_business =self.aggregation_average_sum_curr("business",business2review2word_level_df)
        business_agg_sum_review = self.aggregation_average_sum("business",business2review2word_level_df,"review_id")
        
        id2vec_df = id2vec_df.merge(user_agg_sum_user,on=["user_id"])
        id2vec_df = id2vec_df.merge(user_agg_sum_review,on=["review_id"])
        id2vec_df = self.aggregation_average_divide_curr("user", id2vec_df)
        
        id2vec_df = id2vec_df.merge(business_agg_sum_business, on=["business_id"])
        id2vec_df = id2vec_df.merge(business_agg_sum_review, on=["review_id"])
        id2vec_df = self.aggregation_average_divide_curr("business", id2vec_df)
        
        self.id2vec_df = id2vec_df
        
        logger.info("Division (with curr)")
        print("Division (with curr)")

        
        user_agg_avg_user = self.aggregation_average_divide("user", user_agg_sum_user)
        business_agg_avg_business = self.aggregation_average_divide("business", business_agg_sum_business)

        user2vec_df = user_agg_avg_user[["user_id","user_vec"]]
        self.user2vec_df = user2vec_df
        business2vec_df = business_agg_avg_business[["business_id", "business_vec"]]
        self.business2vec_df = business2vec_df
        
    def build_unseen_samples(self):
        '''
        Creates the unseen samples for the interaction prediction problem dataset.
        It just iterative creates a bundle of random user-business pairs, eliminates already-seen pairs,
        accumulating those left over until the pile roughly meets the same size as observed pairs
        '''
        chunk_size = 5000
        logger.info("Fabricating negative samples")
        print("Fabricating negative samples")

        id2vec_df = self.id2vec_df

        valid_pairs = id2vec_df[["user_id", "business_id"]].reset_index(drop=True)

        num_valid_pairs = valid_pairs.shape[0]

        fraction = (chunk_size/id2vec_df.shape[0])
        user_fraction = fraction
        business_fraction = fraction
        sampled_user_id_column = id2vec_df["user_id"].sample(frac=user_fraction).reset_index(drop=True)
        sampled_business_id_column = id2vec_df["business_id"].sample(frac=business_fraction).reset_index(drop=True)

        sampled_pairs = pd.DataFrame()
        sampled_pairs["user_id"] = sampled_user_id_column.values
        sampled_pairs["business_id"] = sampled_business_id_column.values
        sampled_pairs.reset_index(inplace=True,drop=True)

        sampled_pairs = pd.merge(sampled_pairs, 
                                 valid_pairs, 
                                 on=['user_id', 'business_id'], 
                                 how='outer', 
                                 indicator=True)
        sampled_pairs = sampled_pairs[sampled_pairs['_merge'] == 'left_only']
        sampled_pairs = sampled_pairs.drop(columns=['_merge'])

        while sampled_pairs.shape[0] < num_valid_pairs:
            logger.info("\t", sampled_pairs.shape[0])
            temp_sampled_user_id_column = id2vec_df["user_id"].sample(frac=user_fraction).reset_index(drop=True)
            temp_sampled_business_id_column = id2vec_df["business_id"].sample(frac=business_fraction).reset_index(drop=True)

            temp_sampled_pairs = pd.DataFrame()
            temp_sampled_pairs["user_id"] = temp_sampled_user_id_column.values
            temp_sampled_pairs["business_id"] = temp_sampled_business_id_column.values
            temp_sampled_pairs.reset_index(inplace=True,drop=True)

            temp_sampled_pairs = pd.merge(
                temp_sampled_pairs, 
                valid_pairs, 
                on=['user_id', 'business_id'], 
                how='outer', 
                indicator=True)
            temp_sampled_pairs = temp_sampled_pairs[temp_sampled_pairs['_merge'] == 'left_only']
            temp_sampled_pairs = temp_sampled_pairs.drop(columns=['_merge'])
            sampled_pairs = pd.concat([sampled_pairs,temp_sampled_pairs],axis=0)

        id2vec_df_unseen = sampled_pairs.merge(self.user2vec_df,on=["user_id"])
        id2vec_df_unseen = id2vec_df_unseen.merge(self.business2vec_df,on=["business_id"])
        self.id2vec_df_unseen = id2vec_df_unseen
        return id2vec_df_unseen
    def conjoin_embeddings(self, temp_id2vec_df):
        '''
        Intermediate function that determines how the user and business embedding for a given row should be treated:
        concatenated, or cross multiplied (like a dotproduct that hasn't been summed yet)
        '''
        normalize_vector = lambda x: np.array(x)/np.linalg.norm(np.array(x))
        if self.with_concatenated == 1:
            conjoined_vecs = (
                temp_id2vec_df["user_vec"].apply(list) + temp_id2vec_df["business_vec"].apply(list)
            ).values
        else:
            conjoined_vecs = (
                (temp_id2vec_df["user_vec"].apply(normalize_vector)) * \
                (temp_id2vec_df["business_vec"].apply(normalize_vector))
            )
        return np.array(conjoined_vecs.tolist())
    def set_up(self, reviews_df):
        self.split_df_id_review(reviews_df)
        self.build_review2word()
        #review2word_df = self.build_review2word()
        #self.review2word_df = review2word_df
        
        self.temp_id_removal()
        del self.review_df
        self.build_id2review2word_df()
        #id2review2word_df = self.build_id2review2word_df()
        #self.id2review2word_df = id2review2word_df
        del self.review2word_df
        del self.id_df
        self.build_id2vec_df()
        
        self.build_word_level_dfs()
        #user2review2word_level_df, business2review2word_level_df = self.build_word_level_dfs(id2review2word_df)
        #self.user2review2word_level_df = user2review2word_level_df
        #self.business2review2word_level_df = business2review2word_level_df
        #del self.id2review2word_df
        self.aggregation_averages()
        
        del self.user2review2word_level_df
        del self.business2review2word_level_df
        
        self.build_unseen_samples()
        

        
        
class Temp_Interaction_Data_Preparation_Builder_Pyspark (Temp_Interaction_Data_Preparation_Builder):
    def __init__(
        self,
        recipe_tfidf,
        word2vec_df,
        configs = None,
        spark = None
    ):
        if configs is None:
            with_tfidf_weighting = 1
            with_concatenated = 1
            using_spark = 0
        else:
            with_tfidf_weighting = configs["Temp_Interaction_Data_Preparation_Builder"]["with_tfidf_weighting"]
            with_concatenated = configs["Temp_Interaction_Data_Preparation_Builder"]["with_concatenated"]
            using_spark = configs["Temp_Interaction_Data_Preparation_Builder"]["using_spark"]
        super(Temp_Interaction_Data_Preparation_Builder_Pyspark, self).__init__(
        self,
        recipe_tfidf,
        word2vec_df,
        configs = None,
        spark = None
    ): 
        self.stop_words = stopwords.words("english")
        self.recipe_word2vec_lexicon = recipe_word2vec.wv.vocab
        self.recipe_tfidf_lexicon = list(recipe_tfidf.vocab2idx.keys())
        
        if self.using_spark == 1:
            self.spark = spark
    def split_df_id_review(
        self, 
        reviews_df
    ):
        # ONLY RUN ONCE AT THE START
        # reviews_df contains the y column!
        # reviews_df must be all positives, and before train/test split
        
        new_review_id_count = self.curr_review_id_count + reviews_df.shape[0]


        win_row_number = Window.orderBy("review_id")
        temp_df = reviews_df.withColumn('review_id',F.row_number().over(win_row_number))
        id_df = reviews_df.select(F.col("user_id"),F.col("business_id"),F.col("review_id"))
        review_df = reviews_df.select(F.col("review_id"),F.col("review_text"))

        self.id_df = id_df
        self.review_df = review_df
    def build_review2word(self, review_df=None):
        
        if review_df is None:
            review_df = self.review_df
            
        
        review_df = review_df.withColumn("review_text",F.lower(("review_text")))
        review_df = review_df.withColumn("review_text", F.translate("review_text", non_alphanumeric, " "*len(non_alphanumeric)))
        # https://sparkbyexamples.com/pyspark/pyspark-replace-column-values/
        review_df = review_df.withColumn("review_text", F.split("review_text", " "))
        # https://sparkbyexamples.com/pyspark/pyspark-convert-string-to-array-column/ 
        
        review2word_df = review_df.withColumn("review_text", F.explode("review_text"))

        logger.info("Exploded review_text")
        print("Exploded review_text")
    
        review2word_df = review2word_df.filter(~((F.col("review_text")).isin(self.stop_words)))
        review2word_df = review2word_df.filter(((F.col("review_text")).isin(self.recipe_word2vec_lexicon)))
        review2word_df = review2word_df.filter(((F.col("review_text")).isin(self.recipe_tfidf_lexicon)))
        review2word_df = review2word_df.filter(((F.col("review_text")) != ""))

        logger.info("Filtered exploded review_text")
        print("Filtered exploded review_text")

        self.review2word_df = review2word_df
        return review2word_df
    
    def temp_id_removal(self, 
                        #review2word_df=None
                       ):
        '''
        Removes words that aren't in the tf idf dictionary, or aren't in in the word2vec, or is in stop-words
        Afterwards, it removes any review_id that did not survive said operations
        It also removes review_ids where BOTH the user id and business id only appear 1 one time
        (since when we remove current samples from the averaging later on, it would pretty much remove the whole row)
        '''
        logger.info("temp_id_removal")
        print("temp_id_removal")

        review2word_df = self.review2word_df
        review_df = self.review_df
        id_df = self.id_df
        
        temp_valid_ids_after_exploded_review_text_filter = review2word_df.select(F.col("review_id")).distinct()#.collect()

         # filter out the rows with invalid review ids in old review_df,
        review_df = review_df.join(temp_valid_ids_after_exploded_review_text_filter, on="review_id")
        self.review_df = review_df
        
        # clean up id_df
        # filter out the rows with invalid review ids in old id_df,
        id_df = id_df.merge(temp_valid_ids_after_exploded_review_text_filter, on="review_id")
        # Remove double singletons
        

        business_id_frequency = id_df.groupBy("business_id").count()
        user_id_frequency = id_df.groupBy("user_id").count()

        business_id_one_time = business_id_frequency.filter(((F.col("count"))==1)).select(F.col("business_id"))
        user_id_one_time = user_id_frequency.filter(((F.col("count"))==1)).select(F.col("user_id"))

        id_df = id_df.filter(~(
            ((F.col("business_id").isin(business_id_one_time))) & ((F.col("user_id").isin(user_id_one_time)))
        ))
        self.id_df = id_df
        
        logger.info("Removed double singletons")
        print("Removed double singletons")

    
        # for the word-level, vec can join in now or after tfidf
        # for the word-level, tfidf needs to regroup the level and re-explode it.
        # this re-explosion stage must be separate and after the first explosion, due to the id-removal in between
        # (which also affects the groupby for the user or business ids in the re-explosion)
    
    def _get_tfidfs_sentence(self, tfidf_features_column, row_idx, words):
        return [(tfidf_features_column.iloc[row_idx][self.recipe_tfidf.vocab2idx[word]]) for word in words]
    
    def build_review2word2weight(self, temp_id_column_name, temp_id2review2word_df):
        # TODO: convert this into pyspark format
        '''
        Adds the weighting column (word_count) to the word-wise granularity level table
        This weighting column is how the weighted averages of the vector embeddings will be 
        In the usual circumstance ("unweighted"), the weight column is just the word count itself
        But when weighting is set to true, tf-idf weighting will be used
        '''
        # unless we have a word index column, we'd need the review_id to be exploded
        temp_id2review_df = temp_id2review2word_df.groupBy(temp_id_column_name).agg(
            F.collect_list("review_id"),
            F.collect_list("review_text")
        )
        # https://sparkbyexamples.com/spark/spark-collect-list-and-collect-set-functions/
        
        #temp_vocab = temp_id2review_df.select(F.col()).distinct()
        #temp_vocab = temp_vocab.with

                                               
        cv = M.feature.CountVectorizer(inputCol="review_text", outputCol="count",vocabSize=len(recipe_tfidf_lexicon))
        # https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.CountVectorizer.html
        # https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.IDF.html

        #vectorizer = Pipeline([('count', CountVectorizer(
        #    vocabulary=self.recipe_tfidf.vocab2idx,
        #    stop_words=stop_words,
        #    max_df=1,
        #)), ('tfidf', TfidfTransformer())])
        
        temp_vocab = temp_id2review_df["
        cv_model = cv.fit(temp_id2review_df)
        result = cv_model.transform(temp_id2review_df)
                                       
        # https://stackoverflow.com/questions/59355956/explanation-of-spark-ml-countvectorizer-output
                                       
        
        idf = M.feature.IDF(inputCol="count", outputCol="features")
        # https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.IDF.html
        # https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.IDFModel.html
        # https://stackoverflow.com/questions/59355956/explanation-of-spark-ml-countvectorizer-output

        # https://www.analyticsvidhya.com/blog/2022/09/implementing-count-vectorizer-and-tf-idf-in-nlp-using-pyspark/
        idf_model = idf.fit(result)
        tfidf_features = idf_model.transform(result)
                                       
        # https://stackoverflow.com/questions/50255356/pyspark-countvectorizer-and-word-frequency-in-a-corpus

        # https://spark.apache.org/docs/latest/ml-features#tf-idf
        
        
        #vectorizer.fit(temp_id2review_df['review_text'])
        #tfidf_features = vectorizer.transform(temp_id2review_df["review_text"])##
        
        #assert (temp_id2review_df.shape[0] == tfidf_features.shape[0])
        #temp_id2review_df["review_text"] = temp_id2review_df["review_text"].str.split(" ")
        #tfidf_features_column = pd.Series([tfidf_features[i].A[0] for i in range(tfidf_features.shape[0])])
        #https://stats.stackexchange.com/questions/311260/understanding-and-interpreting-the-output-of-sparks-tf-idf-implementation
        
        

        word_counts_column = [self._get_tfidfs_sentence(tfidf_features_column, i, words) 
                              for i, words 
                              in enumerate(temp_id2review_df["review_text"].values)]

        temp_id2review_df["word_count"] = word_counts_column

        temp_id2review2word2weight_df = temp_id2review_df.explode(["review_id", "review_text","word_count"])
        # now back at the same granularity, and same form as temp_id2review2word_df , but with a new weight column
        return temp_id2review2word2weight_df

    def build_review2word2vec(self, temp_id2review2word_df):
        '''
        Adds the word2vec column to the word-wise granularity level table
        by merging with the word2vec table (which maps words to their vector representations)
        '''
        self.vec_column_names = self.word2vec_df.vec_column_names
        
        temp_id2review2word2vec_df = temp_id2review2word_df.join(self.word2vec_df, on="review_text")
        return temp_id2review2word2vec_df
    
    def build_word_level_df(self, temp_id_column_name, temp_id2review2word_df):
        '''
        This function will create the table for either user2review2word_df or  business2review2word_df
        Both tables are at the word-wise granularity level
        With columns user_id/business_id, review_id, review_text, word_count, and vector embedding
        Where [user_id/business_id, review_id, review_text] form a super key
        Thus each row is a word (and its tf-idf score, and its word2vec vector) of a review, of a user or business id
        '''
        logger.info(temp_id2review2word_df.columns)
        temp_id2review2word2weight_df = self.build_review2word2weight(temp_id_column_name, temp_id2review2word_df)  
        if self.with_tfidf_weighting == 0:
            # undo all tf-idf weighting (treat it as a regular unweighted average)
            temp_id2review2word2weight_df = temp_id2review2word2weight_df.withColumn("word_count", F.lit(1))
            
        temp_id2review2word2weight2vec_df = self.build_review2word2vec(temp_id2review2word2weight_df)
        temp_id2review2word_level_df = temp_id2review2word2weight2vec_df
        return temp_id2review2word_level_df
    
    def build_id2review2word_df(self):
        # ##### word-level is now [user_id, review_id, word, tfidf (vanilla as word_count)]
        review2word_df = self.review2word_df
        id_df = self.id_df
        
        #review2word_df.sort_values(["review_id"], inplace=True)
        id2review2word_df = id_df.join(review2word_df, on="review_id")
        self.id2review2word_df = id2review2word_df
        return id2review2word_df
    def build_id2vec_df(self, id2review2word_df=None):
        '''
        Creates id2vec_df, which has columns ["user_id","business_id","review_id"]
        id2vec_df is an intermediate table from which the tables for 
        mapping user_id / business_id to vector embeddings will be created
        '''
        if id2review2word_df is None:
            id2review2word_df = self.id2review2word_df
        id2vec_df = id2review2word_df.select(F.col("user_id"),F.col("business_id"),F.col("review_id")).distinct()
        # In id2vec_df, review_id work as a unique primary key
        self.id2vec_df = id2vec_df
        return id2vec_df
    def build_word_level_dfs(self, id2review2word_df=None):
        '''
        This function will create the tables user2review2word_df,  business2review2word_df
        Both tables are at the word-wise granularity level
        With columns user_id/business_id, review_id, review_text, word_count, and vector embedding
        Where [user_id/business_id, review_id, review_text] form a super key
        Thus each row is a word (and its tf-idf score, and its word2vec vector) of a review, of a user or business id
        '''
        logger.info("build_word_level_dfs")
        print("build_word_level_dfs")

        if id2review2word_df is None:
            id2review2word_df = self.id2review2word_df
            
        id2review2word_df = id2review2word_df
        user2review2word_df = id2review2word_df.select(F.col("user_id"), F.col("review_id"),F.col("review_text"))
        business2review2word_df = id2review2word_df(F.col("business_id"), F.col("review_id"),F.col("review_text"))

        user2review2word2weight2vec_df = self.build_word_level_df("user_id", user2review2word_df)
        business2review2word2weight2vec_df = self.build_word_level_df("business_id", business2review2word_df)
        
        user2review2word_level_df = user2review2word2weight2vec_df
        business2review2word_level_df = business2review2word2weight2vec_df
        
        self.user2review2word_level_df = user2review2word_level_df
        self.business2review2word_level_df = business2review2word_level_df
        return user2review2word_level_df, business2review2word_level_df

    def aggregation_average_sum(
        self, 
        temp_id_version,
        temp_id2review2word_level_df,
        groupby_column_name,
    ):
        vec_column_names = self.vec_column_names
        agg_column_names = vec_column_names + ["word_count"]
        temp_id_agg = temp_id_agg.groupBy("review_id").sum(agg_column_names)

        return temp_id_agg
    def aggregation_average_sum_curr(
        self, 
        temp_id_version,
        temp_id2review2word_level_df
    ):  
        temp_id_agg = self.aggregation_average_sum(
            temp_id_version, 
            temp_id2review2word_level_df,
            temp_id_version+"_id", 
        )
        vec_column_names = self.vec_column_names
        temp_id_vec_column_names = [temp_id_version+"_"+x for x in vec_column_names]
        for i in range(len(vec_column_names)):
            vec_column_name = vec_column_names[i]
            temp_id_vec_column_name = temp_id_vec_column_name[i]
            temp_id_agg = temp_id_agg.withColumnRenamed(vec_column_name, temp_id_vec_column_name)

        temp_id_agg = temp_id_agg.withColumnRenamed("word_count", temp_id_version+"_word_count")
        return temp_id_agg
        
    def aggregation_average_divide(self, temp_id_version, agg_df):
        
        vec_column_names = self.vec_column_names
        temp_id_vec_column_names = [temp_id_version+"_"+x for x in vec_column_names]
        for i in range(len(vec_column_names)):
            vec_column_name = vec_column_names[i]
            temp_id_vec_column_name = temp_id_vec_column_names[i]
            agg_df = agg_df.withColumn(temp_id_vec_column_name, 
                                             ((F.col(temp_id_vec_column_name)) / (F.col(temp_id_version+"_word_count")))
                                            )
        
        return agg_df

    def aggregation_average_divide_curr(self, temp_id_version, agg_df):

        vec_column_names = self.vec_column_names
        temp_id_vec_column_names = [temp_id_version+"_"+x for x in vec_column_names]
        for i in range(len(vec_column_names)):
            vec_column_name = vec_column_names[i]
            temp_id_vec_column_name = temp_id_vec_column_names[i]
            agg_df = agg_df.withColumn(temp_id_vec_column_name, 
                                             ((F.col(temp_id_vec_column_name)) - (F.col(vec_column_name)))
                                            )
        
        agg_df = agg_df.withColumn(temp_id_version+"_word_count", 
                                  F.col("word_count")
                                  )
        agg_df = agg_df.filter(
            ((F.col(temp_id_version+"_word_count")) != 0)
                              )        
        agg_df = self.aggregation_average_divide(temp_id_version, agg_df)
        agg_df.drop(columns=["vec","word_count"],inplace=True)
        return agg_df
    
    def aggregation_averages(self):
        '''
        This function creates the weighted averages used for summarizing the user id's / business id's
        in terms of their words.
        Rather than directly column-wise averaging the word2vec vectors with an average function,
        this function breaks it apart as 2 separate operations (summing and dividing)
        
        2 reasons for why are:
        * Keeping the operations separated allows to subtract the current review aggregates from the overall review aggregates,
        which is necessary because otherwise the dataset for the prediction will accidentally give away the information. 
        For unobserved pairs, this is impossible to happen and so the regular averages 
        (without current mean subtraction) gets used.
        
        * To make thing extensible to new incoming data
        While simply doing 2 averages (with and without the current running rows) is an option, that feels too bulky
        '''
        logger.info("aggregation_averages")
        print("aggregation_averages")

        id2vec_df = self.id2vec_df
        user2review2word_level_df = self.user2review2word_level_df
        business2review2word_level_df = self.business2review2word_level_df

        logger.info("Aggregation for users, businesses")
        print("Aggregation for users, businesses")

        user_agg_sum_user = self.aggregation_average_sum_curr("user",user2review2word_level_df)
        user_agg_sum_review = self.aggregation_average_sum("user",user2review2word_level_df,"review_id")
        business_agg_sum_business =self.aggregation_average_sum_curr("business",business2review2word_level_df)
        business_agg_sum_review = self.aggregation_average_sum("business",business2review2word_level_df,"review_id")
        
        id2vec_df = id2vec_df.join(user_agg_sum_user,on="user_id")
        id2vec_df = id2vec_df.join(user_agg_sum_review,on="review_id")
        id2vec_df = self.aggregation_average_divide_curr("user", id2vec_df)
        
        id2vec_df = id2vec_df.merge(business_agg_sum_business, on="business_id")
        id2vec_df = id2vec_df.merge(business_agg_sum_review, on="review_id")
        id2vec_df = self.aggregation_average_divide_curr("business", id2vec_df)
        
        self.id2vec_df = id2vec_df
        
        logger.info("Division (with curr)")
        print("Division (with curr)")

        
        user_agg_avg_user = self.aggregation_average_divide("user", user_agg_sum_user)
        business_agg_avg_business = self.aggregation_average_divide("business", business_agg_sum_business)
        
        vec_column_names = self.vec_column_names

        user_vec_column_names = ["user_"+x for x in vec_column_names]
        selected_user_column_names = [F.col(x) for x in ["user_id"] + user_vec_column_names]

        user2vec_df = user_agg_avg_user.select(*selected_user_column_names)
        self.user2vec_df = user2vec_df
        
        business_vec_column_names = ["business_"+x for x in vec_column_names]
        selected_business_column_names = [F.col(x) for x in ["business_id"] + business_vec_column_names]
        
        business2vec_df = business_agg_avg_business.select(*selected_business_column_names)
        self.business2vec_df = business2vec_df
        
    def build_unseen_samples(self):
        '''
        Creates the unseen samples for the interaction prediction problem dataset.
        It just iterative creates a bundle of random user-business pairs, eliminates already-seen pairs,
        accumulating those left over until the pile roughly meets the same size as observed pairs
        '''
        chunk_size = 5000
        logger.info("Fabricating negative samples")
        print("Fabricating negative samples")

        id2vec_df = self.id2vec_df

        valid_pairs = id2vec_df.select(F.col("user_id"), F.col("business_id"))

        num_valid_pairs = valid_pairs.count()

        fraction = (chunk_size/num_valid_pairs)
        user_fraction = fraction
        business_fraction = fraction
        
        valid_user_id_column = id2vec_df.select(F.col("user_id"))
        valid_business_id_column = id2vec_df.select(F.col("business_id"))

        temp_user_id_samples = valid_user_id_column.sample(fraction=1)
        temp_business_id_samples = valid_business_id_column.sample(fraction=1)
        
        sampled_user_id_column = temp_user_id_samples.sample(frac=user_fraction)
        sampled_business_id_column = temp_business_id_samples.sample(frac=business_fraction)

        sampled_pairs = sampled_user_id_column
        sampled_pairs = sampled_pairs.withColumn("business_id", 
                                                 sampled_business_id_column.select(F.col("business_id"))
                                                )
        unseen_pairs = sampled_pairs.difference(valid_pairs)

        while unseen_pairs.count() < num_valid_pairs:
            logger.info("\t", sampled_pairs.count())
            
            #print("\t", sampled_pairs.count())
            

            temp_user_id_samples = valid_user_id_column.sample(fraction=1)
            temp_business_id_samples = valid_business_id_column.sample(fraction=1)

            temp_user_id_sampled = temp_user_id_samples.sample(fraction=user_fraction)
            #temp_user_id_sampled = temp_user_id_sampled.limit(chunk_size)
            temp_business_id_sampled = temp_business_id_samples.sample(fraction=business_fraction)
            #temp_business_id_sampled = temp_business_id_sampled.limit(chunk_size)

            temp_sampled_pairs = temp_user_id_sampled.withColumn("business_id", 
                                                                   temp_business_id_sampled.select(F.col("business_id"))
                                                                  )
            temp_unseen_pairs = temp_sampled_pairs.difference(valid_pairs)
            unseen_pairs = unseen_pairs.union(temp_unseen_pairs)
            
            
        id2vec_df_unseen = unseen_pairs

        id2vec_df_unseen = id2vec_df_unseen.join(self.user2vec_df,on="user_id")
        id2vec_df_unseen = id2vec_df_unseen.join(self.business2vec_df,on="business_id")
        self.id2vec_df_unseen = id2vec_df_unseen
        return id2vec_df_unseen
    def conjoin_embeddings(self, temp_id2vec_df):
        '''
        Intermediate function that determines how the user and business embedding for a given row should be treated:
        concatenated, or cross multiplied (like a dotproduct that hasn't been summed yet)
        '''
        vec_column_names = self.vec_column_names
        user_vec_column_names = ["user_"+x for x in vec_column_names]
        business_vec_column_names = ["business_"+x for x in vec_column_names]
        
        normalize_vector = lambda x: np.array(x)/np.linalg.norm(np.array(x))
        if self.with_concatenated == 1:
            conjoined_vecs_selected_column_names = [F.col(x) for x in user_vec_column_names+business_vec_column_names]
            conjoined_vecs = (
                temp_id2vec_df.select(*conjoined_vecs_selected_column_names)
            ).values
        else:
            for temp_id_version in ["user","business"]:
                temp_id_vec_norm_column_name = temp_id_version+"_vec_norm"
                temp_id2vec_df = temp_id2vec_df.withColumn(temp_id_vec_norm_column_name, F.lit(0))
                for i in range(len(vec_column_names)):
                    vec_column_name = vec_column_names[i]
                    temp_id_vec_column_name = temp_id_version + "_" + vec_column_name
                    temp_id2vec_df = temp_id2vec_df.withColumn(
                        temp_id_vec_norm_column_name,
                        F.col(temp_id_vec_norm_column_name) + F.pow(F.col(temp_id_vec_column_name),2)
                    )
                temp_id2vec_df = temp_id2vec_df.withColumn(temp_id_vec_norm_column_name, 
                                                           F.sqrt(F.col(temp_id_vec_norm_column_name))
                                                          )
                
                for i in range(len(vec_column_names)):
                    vec_column_name = vec_column_names[i]
                    temp_id_vec_column_name = temp_id_version + "_" + vec_column_name
                    temp_id2vec_df = temp_id2vec_df.withColumn(
                        temp_id_vec_column_name,
                        F.col(temp_id_vec_column_name) / F.col(temp_id_vec_norm_column_name)
                    )
            
                
            for i in range(len(vec_column_names)):
                vec_column_name = vec_column_names[i]
                user_vec_column_name = "user_" + vec_column_name
                business_vec_column_name = "business_" + vec_column_name
                temp_id2vec_df = temp_id2vec_df.withColumn(
                    vec_column_name,
                    F.col(user_vec_column_name) * F.col(business_vec_column_name)
                )
            conjoined_vecs_selected_column_names = [F.col(x) for x in vec_column_name]
            conjoined_vecs = temp_id2vec_df.select(*conjoined_vecs_selected_column_names)
        return conjoined_vecs
    def set_up(self, reviews_df):
        self.split_df_id_review(reviews_df)
        self.build_review2word()
        #review2word_df = self.build_review2word()
        #self.review2word_df = review2word_df
        
        self.temp_id_removal()
        del self.review_df
        self.build_id2review2word_df()
        #id2review2word_df = self.build_id2review2word_df()
        #self.id2review2word_df = id2review2word_df
        del self.review2word_df
        del self.id_df
        self.build_id2vec_df()
        
        self.build_word_level_dfs()
        #user2review2word_level_df, business2review2word_level_df = self.build_word_level_dfs(id2review2word_df)
        #self.user2review2word_level_df = user2review2word_level_df
        #self.business2review2word_level_df = business2review2word_level_df
        #del self.id2review2word_df
        self.aggregation_averages()
        
        del self.user2review2word_level_df
        del self.business2review2word_level_df
        
        self.build_unseen_samples()
        
