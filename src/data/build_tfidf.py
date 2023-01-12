import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
path_folder = (os.path.abspath(os.path.join((os.path.abspath(os.path.join(os.getcwd(), os.pardir))),os.pardir)))

google_colab = 0
if google_colab == 1:
    from google.colab import drive
    drive.mount('/content/drive/')
    path_folder = "/content/drive/MyDrive/dsprojects/dsproject_grev/"
    
sys.path.insert(0, path_folder+"/src/"#+features/"
                )
import util

import nltk
from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
#from nltk.tokenize import word_tokenize
nltk.download('stopwords')
stop_words = stopwords.words('english')

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def save(path_file,tfidf_model):
    with open(path_file, "wb") as f:
        pickle.dump(tfidf_model, f)
        f.close()
    return tfidf_model
def load(path_file):
    with open(path_file, "rb") as f:
        tfidf_model = pickle.load(f)
        f.close()
    return tfidf_model
def main(configs=None, method=None):
    # args:
    #vocab_column_names = tfidf_vocab_column_names
    if configs is None:
        path_folder = "../../data/"
        vocab_column_names = [
            "ingredients",
            #"name",
            ]
        max_df = 1.0
        min_df = 1
        #stop_words = stopwords.words("english")
        vocabulary = None
    else:
        path_folder = configs["build_tfidf"]["path_folder"]
        vocab_column_names = configs["build_tfidf"]["vocab_column_names"]
        max_df = configs["build_tfidf"]["max_df"]
        min_df = configs["build_tfidf"]["min_df"]
        #stop_words = stopwords.words("english")
        vocabulary = configs["build_tfidf"]["vocabulary"]
    method = method
    path_folder = path_folder
    valid_vocab_column_names = ["tags","ingredients","name","steps","description"]
    assert all((arg in valid_vocab_column_names) for arg in vocab_column_names)
    path_data_raw_recipes =  os.path.join(path_folder,"raw","RAW_recipes.csv")
    raw_recipes_df = pd.read_csv(path_data_raw_recipes)
    file_name = "tfidf_"+("_".join(vocab_column_names))+".pkl"
    path_file = os.path.join(path_folder, "temp", file_name)

    if method is "load":
        return load(path_file)

    liststr_columns = util.get_liststr_columns(raw_recipes_df, vocab_column_names)
    food_vocab_column = (liststr_columns).apply(lambda x: " ".join(x))
    
    vectorizer = Pipeline(
        steps=[
            ('count',
             CountVectorizer(
                 stop_words=stop_words,
                 max_df=max_df,
                 min_df=min_df,
                 vocabulary=vocabulary,
                 )),
            ('tfidf',
             TfidfTransformer(
                 ))
            ]
        )
    tfidf_model = util.Temp_Recipe_Vectorizer_Builder(vectorizer, food_vocab_column)

    if method is "save":
        save(path_file, tfidf_model)
    return tfidf_model
# params
# vocab_columns
# > file_name
# tfidf args






