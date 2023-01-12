#!/usr/bin/env python
# coding: utf-8

# In[2]:


configs = {
    "Reviews_Dataset_Reader": {
        "path_folder": "../../data/",
        "dataset_version": "FILTER",
        "using_pics": 0,
        "using_spark": 0,
        "pic_link_header": "https://lh5.googleusercontent.com/p/",
    },
    "Word2vec_Dataset_Reader": {
        "vector_mode": "COMPRESSED",
        "using_spark": 0,
    },
    "build_tfidf": {
        "path_folder": "../../data/",
        "vocab_column_names": ["ingredients"],
        "max_df": 1.0,
        "min_df": 1,
        "vocabulary": None,
    },
    "build_word2vec": {
        "path_folder": "../../data/",
        "vocab_column_names": ["ingredients","name"],
        "seed": 1,
        "size": 100,
        "window": 5,
        "min_count": 1,
    },
    "Temp_Interaction_Data_Preparation_Builder": {
            "with_tfidf_weighting": 1,
            "with_concatenated": 1,
            "using_spark": 0,
    },
}


# In[6]:


import json
with open("configs.json","w") as f:
    json.dump(configs, f, indent = 6)
    f.close()


# In[ ]:




