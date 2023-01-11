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
    
sys.path.insert(0, path_folder+"/src/features/")
import util

from word2vec_recipe import *

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

import string

USING_SPARK = 0
SPARK_SESSION_VERSION = 1
if USING_SPARK == 1:
    import findspark
    import pyspark
    import pyspark.sql.functions as F
    import pyspark.sql.types as T
    import pyspark.ml as M
    from pyspark.sql.window import Window
    from pyspark.sql import SparkSession
    findspark.init()
    if SPARK_SESSION_VERSION == 1:
        spark = SparkSession.builder.getOrCreate()
    elif SPARK_SESSION_VERSION == 2:
        spark = SparkSession.builder             .master('local[*]')             .config("spark.driver.memory", "3g")             .appName('food_rec')             .getOrCreate()
    

class Temp_Interaction_Model_Builder:
    def __init__(
        self,
        recipe_tfidf,
        word2vec_df,
        with_tfidf_weighting = 1,
        with_concatenated = 1
    ):
        self.recipe_tfidf = recipe_tfidf
        self.word2vec_df = word2vec_df
        self.with_tfidf_weighting = with_tfidf_weighting
        self.with_concatenated = with_concatenated
        self.id_df = pd.DataFrame(columns=["user_id","business_id","review_id"])
        self.review_df = pd.DataFrame(columns=["review_id","review_text"])
        self.curr_review_id_count = 0
    def update_id_df(self, id_df):
        new_id_df = pd.concat([self.id_df, id_df],axis=0)
        self.id_df = new_id_df
        #return new_id_df
    def update_review_df(self, review_df):
        new_review_df = pd.concat([self.review_df, review_df],axis=0) 
        self.review_df = new_review_df
        #return new_review_df
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
        review2word_df = review2word_df[~(review2word_df["review_text"].isin(stop_words))]
        review2word_df = review2word_df[(review2word_df["review_text"].isin(recipe_word2vec.wv.vocab))]
        review2word_df = review2word_df[(review2word_df["review_text"].isin(recipe_tfidf.vocab2idx))]
        review2word_df = review2word_df[(review2word_df["review_text"] != "")]
        logger.info("Filtered exploded review_text")
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
        '''
        
        '''
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
        id2vec_df = self.id2vec_df
        user2review2word_level_df = self.user2review2word_level_df
        business2review2word_level_df = self.business2review2word_level_df

        logger.info("Aggregation for users, businesses")
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
        del self.id2review2word_df
        self.aggregation_averages()
        
        del self.user2review2word_level_df
        del self.business2review2word_level_df
        
        self.build_unseen_samples()
    def data_preparation(self, reviews_df):
        self.set_up(reviews_df)
        
        X_seen = self.conjoin_embeddings(self.id2vec_df)
        y_seen = np.ones(X_seen.shape[0])
    
        X_unseen = self.conjoin_embeddings(self.id2vec_df_unseen)
        y_unseen = np.zeros(X_unseen.shape[0])

        train_dataset_size = len(y_seen)+len(y_unseen)
        train_dataset_indexer = np.arange(train_dataset_size)
        np.random.shuffle(train_dataset_indexer)

        X = np.vstack((X_seen,X_unseen))
        y = np.hstack((y_seen,y_unseen))
        X = X[train_dataset_indexer]
        y = y[train_dataset_indexer]
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)
        
        pca = PCA(2)
        temp_X_pca_input = X_train
        temp_y_pca_input = y_train

        Z = pca.fit_transform(temp_X_pca_input)
        fig, ax = plt.subplots(figsize=(8,8))
        ax.scatter(Z[:,0], Z[:,1],c=temp_y_pca_input, alpha=0.1, s=5, cmap="winter")
        
        return X_train, X_val, y_train, y_val


def main():
    vocab_column_names = ["ingredients"]+["name"]
    file_name_word2vec = "word2vec_"+("_".join(vocab_column_names))+".pkl"
    path_file_word2vec = os.path.join(path_folder, "data","temp",file_name_word2vec)
    with open(path_file_word2vec, "rb") as f:
        recipe_word2vec = pickle.load(f)

    vocab_column_names = ["ingredients"]#+["name"]
    file_name_tfidf = "tfidf_"+("_".join(vocab_column_names))+".pkl"
    path_file_tfidf = os.path.join(path_folder, "data","temp",file_name_tfidf)
    with open(path_file_tfidf, "rb") as f:
        recipe_tfidf = pickle.load(f)

    DATASET_VERSION = ["FULL", "FILTER"][1]  
    DROP_ONE_TIME_APPEARANCES = 1
    DROP_SEPARATED_ONE_TIME_APPEARANCES = 0
    USING_PICS = 0

    reviews_dataset_reader = util.Reviews_Dataset_Reader(
        path_folder, 
        DATASET_VERSION, 
        USING_PICS,
        USING_SPARK,
        DROP_ONE_TIME_APPEARANCES, 
        DROP_SEPARATED_ONE_TIME_APPEARANCES,
    )

    reviews_df = reviews_dataset_reader.build_reviews_df()

    stop_words = stopwords.words('english')
    lexicon = [word for word in recipe_tfidf.vocab2idx if word not in stop_words]

    word2vec_dataset_reader = util.Word2vec_Dataset_Reader(
        recipe_word2vec,
        lexicon,
    )
    word2vec_df = word2vec_dataset_reader.build_word2vec_df()

    temp_interaction_model_builder = Temp_Interaction_Model_Builder(
        recipe_tfidf,
        word2vec_df,
        with_concatenated = 0
    )

    X_train, X_val, y_train, y_val = temp_interaction_model_builder.data_preparation(reviews_df)

    log_reg = LogisticRegression(max_iter=800, class_weight="balanced")
    log_reg.fit(X_train, y_train)
    lr_y_pred = log_reg.predict(X_val)

    y_true = y_val
    y_pred = lr_y_pred
    conf_matrix = confusion_matrix(y_true, y_pred,normalize="true")
    fig, ax2 = plt.subplots(figsize=(10,10)) 
    ax2 = sns.heatmap(conf_matrix, annot=True) #notation: "annot" not "annote"
    y_lims = ax2.get_ylim()
    ax2.set_ylim(sum(y_lims), 0)

    from sklearn.metrics import roc_auc_score
    print("AUC:", roc_auc_score(y_pred.astype(int), y_true.astype(int)))

