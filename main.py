import os
from os import listdir
from os.path import isfile, join
from tempfile import gettempdir
from collections import OrderedDict
from clearml import Task, Dataset,Logger

PROJECT_NAME = "bertopic"
TASK_NAME = "bertopic recursive"

Task.force_requirements_env_freeze(force=True, requirements_file='requirements.txt')
task = Task.init(project_name=PROJECT_NAME, task_name=TASK_NAME)
# task.set_base_docker("nvcr.io/nvidia/pytorch:20.08-py3")
task.set_base_docker("nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04")

args = {'project_name':PROJECT_NAME,'task_name':TASK_NAME}
task.connect(args)

task.execute_remotely()

logger = task.get_logger()

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
import hdbscan

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from typing import List
import string
from string import digits

from pandas import DataFrame, Series

import re
# spacy for lemmatization
import spacy
import umap
import numpy as np

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
nltk.download('stopwords')
nltk.download('punkt')

from scipy_reduce import LevelClusterer 

def add_stopwords(text_data):
    count_vectorizer = CountVectorizer(stop_words='english')
    vectorized = count_vectorizer.fit_transform(text_data)
    vectorized_array = vectorized.toarray()
    total_docs = vectorized_array.shape[0]
    total_vectors = vectorized_array.shape[1]

    doc_count=0
    words = []

    for i in range(0,total_vectors):
        for j in range(0,total_docs):
            if vectorized_array[j][i] > 0:
                doc_count += 1

        percentage_docs = float(doc_count)/float(total_docs)
        if percentage_docs >= 0.6:
            print(percentage_docs)
            print(count_vectorizer.get_feature_names()[i])
            words.append(count_vectorizer.get_feature_names()[i])
        doc_count=0

    return words
    
def remove_stopwords_series(series: Series):

    additional_stop_words = add_stopwords(series)

    # print("Additonal stopwords for data column(above 60% threshold): ")
    # print(additional_stop_words)

    cleaned_series = series.apply(remove_stopwords, additional_stop_words=additional_stop_words)

    return cleaned_series

def remove_stopwords(row: str, additional_stop_words=[]):
    removed_numbers = re.sub(r'[0-9]+', ' ', row)
    no_punc = re.sub(r'[^\w\s]', ' ', removed_numbers)
    # print(no_punc)
    # print('\n')
    sentence_words = nltk.word_tokenize(no_punc)
    stop_words = nltk.corpus.stopwords.words('english')
    stop_words.extend(['the','The'])
    stop_words.extend(additional_stop_words)
    words = [word for word in sentence_words if word not in stop_words and len(word)>2]

    return " ".join(words)

def lemmatization_series(texts, unallowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    
    """https://spacy.io/api/annotation"""
    nlp = spacy.load('en_core_web_sm',disable=["tagger","parser"])
    
    texts = texts.tolist()
    lst = []
    for doc in nlp.pipe(texts,n_process=2,batch_size=1000):
      words = [token.text for token in doc if token.pos_ not in unallowed_postags and token.lemma_!='-PRON-']
      lst.append(" ".join(words))
    cleaned_series = pd.Series(lst)

    return cleaned_series

def lemmatization(text, nlp, unallowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    
    # text = gensim.utils.simple_preprocess(str(text), deacc=True)
    # doc = nlp(" ".join(text)) 
    doc = nlp(text)
    words = [token.text for token in doc if token.pos_ not in unallowed_postags and token.lemma_!='-PRON-']
    
    return " ".join(words)

def clean_urls(review):
    review = review.split()
    review = ' '.join([word for word in review if not re.match('^http', word)])
    return review

def decontracted(text):
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"it\'s", "it is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)

    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    text = re.sub(r"\'m", " am", text)
    text = re.sub(r"\“", "", text)
    text = re.sub(r"\”", "", text)
    text = re.sub(r"\…", "", text)

    return text

def clean_text(text):
    text = str(text)
    text = re.sub(r'(\w)\1{2,}', r'\1', text)
    text = re.sub(r'[^a-zA-Z ]+', ' ', text) #remove any occurance of one or more characters not in a-z or A-Z
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    text = re.sub(r'^RT[\s]+', '', text)
    text = re.sub(r'FleetMon', '', text)
    text = re.sub(r'fleetmon', '', text)
    text = re.sub(r'#', '', text)
    # text = text.lower()

    return text

def _auto_reduce_topics(model, documents: pd.DataFrame) -> pd.DataFrame:
    """ Reduce the number of topics automatically using HDBSCAN
    Arguments:
        documents: Dataframe with documents and their corresponding IDs and Topics
    Returns:
        documents: Updated dataframe with documents and the reduced number of Topics
    """
    topics = documents.Topic.tolist().copy()
    unique_topics = sorted(list(documents.Topic.unique()))[1:]
    max_topic = unique_topics[-1]

    # Find similar topics
    if model.topic_embeddings is not None:
        embeddings = np.array(model.topic_embeddings)
    else:
        embeddings = model.c_tf_idf.toarray()
    norm_data = normalize(embeddings, norm='l2')
    predictions = hdbscan.HDBSCAN(min_cluster_size=2,
                                  metric='euclidean',
                                  cluster_selection_method='leaf',
                                  prediction_data=True).fit_predict(norm_data[1:])

    # Map similar topics
    mapped_topics = {unique_topics[index]: prediction + max_topic
                      for index, prediction in enumerate(predictions)
                      if prediction != -1}
    documents.Topic = documents.Topic.map(mapped_topics).fillna(documents.Topic).astype(int)
    mapped_topics = {from_topic: to_topic for from_topic, to_topic in zip(topics, documents.Topic.tolist())}

    # Update documents and topics
    model.topic_mapper.add_mappings(mapped_topics)
    documents = model._sort_mappings_by_frequency(documents)
    model._extract_topics(documents)
    model._update_topic_size(documents)
    return documents

# get uploaded dataset from clearML
dataset_dict = Dataset.list_datasets(
    dataset_project='datasets/bertopic', partial_name='300 data', only_completed=False
)

datasets_obj = [
    Dataset.get(dataset_id=dataset_dict["id"]) for dataset_dict in dataset_dict
]

# reverse list due to child-parent dependency, and get the first dataset_obj
dataset_obj = datasets_obj[::-1][0]

folder = dataset_obj.get_local_copy()

file = [file for file in dataset_obj.list_files() if file=='300_texts_split.csv'][0]

file_path = folder + "/" + file
df = pd.read_csv(file_path) 
df = df.drop(['SOURCE'], axis=1) 
print(df.head())

nan_value = float("NaN")
df.replace("", nan_value, inplace=True)
df.dropna(axis=0, inplace=True)

df['texts_cleaned'] = df['texts'].apply(clean_urls).apply(clean_text).apply(decontracted)
texts = df['texts_cleaned']
# lemmatize sentences and remove words corresponding to the listed Spacy POS tags in unallowed_postags
texts = lemmatization_series(texts, unallowed_postags=['X', 'SYM', 'PUNCT', 'NUM','SPACE','PROPN']) #,'INTJ','PROPN'

# df.insert(len(df.columns), 'cleaned_texts', clean_col)
df.insert(len(df.columns), 'cleaned_texts', texts)
# nan_value = float("NaN")
# df.replace("", nan_value, inplace=True)
# df.dropna(axis=0, inplace=True)
df.replace("", "Text not suitable for modeling after cleaning", inplace=True)

df=df.drop(['texts_cleaned'], axis=1)
# df = df.drop_duplicates(subset='cleaned_texts', keep="first")
print(df.head())
print(df.info())

docs = df['cleaned_texts'].tolist()

#use Sentence Transformer from Huggingface library to create embeddings of cleaned text to be used as features for topic modelling

# sentence_model = SentenceTransformer("distilbert-base-nli-mean-tokens")
sentence_model = SentenceTransformer("all-mpnet-base-v2")
# sentence_model = SentenceTransformer("all-distilroberta-v1")
embeddings = sentence_model.encode(docs)

umap_model = umap.UMAP(n_neighbors=15,
              n_components=300,
              min_dist=0.0,
              metric='cosine',
              random_state=42)

hdb_model = hdbscan.HDBSCAN(min_cluster_size=10,
                            metric='euclidean',
                            cluster_selection_method='leaf',
                            prediction_data=True)


model = BERTopic(calculate_probabilities=True,verbose=True,umap_model=umap_model,hdbscan_model=hdb_model,embedding_model=sentence_model) #nr_topics="auto",
# model = BERTopic(calculate_probabilities=True,verbose=True,umap_model=umap_model,embedding_model=sentence_model) 
topics, probabilities = model.fit_transform(docs, embeddings)

cluster = topics

nlp = spacy.load("en_core_web_sm")

topics = {}
topic_names = []

for key,value in model.get_topics().items():
  topic_words = [i[0] for i in value]

  unallowed_postags=['X', 'SYM', 'PUNCT', 'NUM','SPACE','ADP','AUX','CONJ','CCONJ','DET','PRON','SCONJ']
  doc = nlp(' '.join(topic_words))
  words = [token.text for token in doc if token.pos_ not in unallowed_postags and token.lemma_!='-PRON-']
    
  if len(words)>=5:
    topics[str(key)] = " ".join(words[:5])
  else:
    topics[str(key)] = " ".join(topic_words[:5])
  topic_names.append(" ".join(topic_words[:5]))

df['topic_number'] = cluster
df=df.sort_values(['topic_number'], ascending=True)

cluster_names_column = pd.Series(df['topic_number'].values).apply(lambda x: topics[str(x)])

df['topic_name'] = cluster_names_column.values
print(df.head(10))
Logger.current_logger().report_table(title='results',series='pandas DataFrame',iteration=0,table_plot=df)
Logger.current_logger().report_plotly(title='level_0',series='Dendrogram',figure=model.visualize_hierarchy(),iteration=0)

df = df.rename(columns={'topic_number':'topic_number_1','topic_name':'topic_name_1'})

level_clusterer = LevelClusterer(df,text_col='cleaned_texts',BERTopic_model=model)

level_clusterer.initial_ranking()
distance_matrix = level_clusterer.calculate_distance_matrix()
cutree = level_clusterer.calculate_cutree()
print("Cutree for topics: ",cutree)
mapping_df = pd.DataFrame(cutree,columns=['level_{}'.format(str(i)) for i in range(level_clusterer.levels)])
Logger.current_logger().report_table(title='cutree levels',series='pandas DataFrame',iteration=0,table_plot=mapping_df)
for level in range(1,level_clusterer.levels):
    fig,results_df = level_clusterer.cut_at_level(level)
    # fig.write_image(os.path.join(gettempdir(), "dendrogram_level_{}.png".format(str(level))))
    Logger.current_logger().report_table(title='leveling results',series='pandas DataFrame',iteration=0,table_plot=results_df)
    Logger.current_logger().report_plotly(title='level_{}'.format(level),series='Dendrogram',figure=fig,iteration=0)


# df['Topic'] = df['topic_number']
# df = df.rename(columns={'cleaned_texts':'Document'})
# df = _auto_reduce_topics(model,df)

# topics = {}
# topic_names = []

# for key,value in model.get_topics().items():
#   topic_words = [i[0] for i in value]

#   unallowed_postags=['X', 'SYM', 'PUNCT', 'NUM','SPACE','ADP','AUX','CONJ','CCONJ','DET','PRON','SCONJ']
#   doc = nlp(' '.join(topic_words))
#   words = [token.text for token in doc if token.pos_ not in unallowed_postags and token.lemma_!='-PRON-']
    
#   if len(words)>=5:
#     topics[str(key)] = " ".join(words[:5])
#   else:
#     topics[str(key)] = " ".join(topic_words[:5])
#   topic_names.append(" ".join(topic_words[:5]))

# cluster_names_column = pd.Series(df['Topic'].values).apply(lambda x: topics[str(x)])
# df['topic_name_2'] = cluster_names_column.values
# df=df.sort_values(['topic_number'], ascending=True)
# print(df.head(10))
# Logger.current_logger().report_table(title='topic_name_2 results',series='pandas DataFrame',iteration=1,table_plot=df)

# rename_count = 2
# while len(model.get_topic_freq()) - 1 > 100:
#     df = df.rename(columns={'Topic':'topic_number_{}'.format(rename_count)})
#     df['Topic'] = df['topic_number_{}'.format(rename_count)]
#     df = df.rename(columns={'cleaned_texts':'Document'})
#     df = _auto_reduce_topics(model,df)
#     rename_count += 1
#     topics = {}
#     topic_names = []

#     for key,value in model.get_topics().items():
#         topic_words = [i[0] for i in value]

#         unallowed_postags=['X', 'SYM', 'PUNCT', 'NUM','SPACE','ADP','AUX','CONJ','CCONJ','DET','PRON','SCONJ']
#         doc = nlp(' '.join(topic_words))
#         words = [token.text for token in doc if token.pos_ not in unallowed_postags and token.lemma_!='-PRON-']

#         if len(words)>=5:
#             topics[str(key)] = " ".join(words[:5])
#         else:
#             topics[str(key)] = " ".join(topic_words[:5])
#         topic_names.append(" ".join(topic_words[:5]))

#     cluster_names_column = pd.Series(df['Topic'].values).apply(lambda x: topics[str(x)])
#     df['topic_name_{}'.format(rename_count)] = cluster_names_column.values
#     df=df.sort_values(['topic_number'], ascending=True)
#     print(df.head(10))
#     Logger.current_logger().report_table(title='topic_name_{} results'.format(rename_count),series='pandas DataFrame',iteration=rename_count,table_plot=df)


# df.to_csv('multi_reduce_df.csv',index=False)
mapping_df.to_csv(os.path.join(gettempdir(), 'mappings_df.csv'),index=False)
results_df.to_csv(os.path.join(gettempdir(), 'multi_reduce_df.csv'),index=False)

dataset = Dataset.create('300 results with mappings 2','datasets/bertopic')

files = [f for f in listdir(gettempdir()) if isfile(join(gettempdir(), f)) and (f.endswith('.csv') or f.endswith('.png'))]

for file in files:
    dataset.add_files(os.path.join(gettempdir(), file))

dataset.upload(output_url='s3://experiment-logging/multimodal')
dataset.finalize()