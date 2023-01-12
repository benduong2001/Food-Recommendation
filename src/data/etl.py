#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import gdown
import pandas as pd
#import requests



def download_csv_from_google_drive_share_link_to_file_path(url, filepath):
    gdown.download(url=url, output=filepath, quiet=False, fuzzy=True)
def download_json_from_google_drive_share_link_to_file_path(url, filepath):
    gdown.download(url=url, output=filepath, quiet=False, fuzzy=True)
def get_file_path(path_folder, filename):
    path_file = os.path.join(path_folder, "raw",filename)
    return path_file
    
def main(path_folder = "../../data/"):
    
    url_google_drive_filter_all_t_json = (
        "https://drive.google.com/file/d/1JH3c83Ac01xkGeM_86lEFO4M8YUps6CS/view?usp=sharing"
    )
    file_name_filter_all_t_json = "filter_all_t.json"

    url_google_drive_image_review_all_json = (
        "https://drive.google.com/file/d/1jg46fLeI4lGgvli4iXBOXgi7ns8jKepH/view?usp=sharing"
    )
    file_name_image_review_all_json = "image_review_all.json"

    url_google_drive_raw_recipes_csv = (
        "https://drive.google.com/file/d/1CovmXQpbPS-slgdYMeh0PGL0yQ2fblgt/view?usp=sharing"
    )
    file_name_raw_recipes_csv = "RAW_recipes.csv"
    
    path_file_filter_all_t_json = get_file_path(path_folder, file_name_filter_all_t_json)
    path_file_image_review_all_json = get_file_path(path_folder, file_name_image_review_all_json)
    path_file_raw_recipes_csv = get_file_path(path_folder, file_name_raw_recipes_csv)
    
    #print(path_file_filter_all_t_json)
    #print(path_file_image_review_all_json)
    #print(path_file_raw_recipes_csv)

    download_json_from_google_drive_share_link_to_file_path(
        url_google_drive_filter_all_t_json, path_file_filter_all_t_json)
    download_json_from_google_drive_share_link_to_file_path(
        url_google_drive_image_review_all_json, path_file_image_review_all_json)
    download_csv_from_google_drive_share_link_to_file_path(
        url_google_drive_raw_recipes_csv, path_file_raw_recipes_csv)
    



