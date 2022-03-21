import os
from os import listdir
from os.path import isfile, join
from tempfile import gettempdir
from clearml import Task, Dataset,Logger

import pandas as pd

dataset_dict = Dataset.list_datasets(
    dataset_project='c4 results', partial_name='datasets/bertopic', only_completed=False
)

datasets_obj = [
    Dataset.get(dataset_id=dataset_dict["id"]) for dataset_dict in dataset_dict
]

# reverse list due to child-parent dependency, and get the first dataset_obj
dataset_obj = datasets_obj[::-1][0]

folder = dataset_obj.get_local_copy()

file = [file for file in dataset_obj.list_files() if file=='multi_reduce_df.csv'][0]

file_path = folder + "/" + file
df = pd.read_csv(file_path)  
print(df.head())
df.to_csv('multi_reduce_df.csv',index_label=False)