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

def plot(word2vec_model):
    outp = util.word2vec_arithmetic_data(word2vec_model, "india","naan bread","arabic" )
    print(outp[::-1])
    outp = util.word2vec_arithmetic_data(word2vec_model, "july 4th", "barbeque", "thanksgiving dinner")
    print(outp[::-1])

def save(path_file, word2vec_model):
    with open(path_file, "wb") as f:
        pickle.dump(word2vec_model, f)
        f.close()
    return word2vec_model
def load(path_file):
    with open(path_file, "rb") as f:
        word2vec_model = pickle.load(f)
        f.close()
    return word2vec_model
def main(configs = None, method=None):
    # args:
    #vocab_column_names = word2vec_vocab_column_names
    if configs is None:
        path_folder="../../data/"
        vocab_column_names = [
            "ingredients",
            "name",
            ]
        seed = 1
        size = 100
        window = 3
        min_count = 1
    else:
        path_folder = configs["build_word2vec"]["path_folder"]
        vocab_column_names = configs["build_word2vec"]["vocab_column_names"]
        seed = configs["build_word2vec"]["seed"]
        size = configs["build_word2vec"]["size"]
        window = configs["build_word2vec"]["window"]
        min_count = configs["build_word2vec"]["min_count"]

    method = method
    
    path_folder = path_folder
    valid_vocab_column_names = ["tags","ingredients","name","steps","description"]
    assert all((arg in valid_vocab_column_names) for arg in vocab_column_names)
    path_data_raw_recipes =  os.path.join(path_folder,"raw","RAW_recipes.csv")
    raw_recipes_df = pd.read_csv(path_data_raw_recipes)
    file_name = "word2vec_"+("_".join(vocab_column_names))+".pkl"
    path_file = os.path.join(path_folder, "temp", file_name)

    if method is "load":
        return load(path_file)

    liststr_columns = util.get_liststr_columns(raw_recipes_df, vocab_column_names)

    np.random.seed(seed)
    word2vec_model = util.dataclean_word2vec(liststr_columns,
                                             size = size,
                                             window = window,
                                             min_count = min_count
                                             )
    plot(word2vec_model)

    if method is "save":
        save(path_file, word2vec_model)
    return word2vec_model


