#from aitextgen.TokenDataset import TokenDataset
#from aitextgen.tokenizers import train_tokenizer
#from aitextgen.utils import GPT2ConfigCPU
from aitextgen import aitextgen
import pickle
import pandas as pd
import imdb
import warnings
import webbrowser
from sentence_transformers import SentenceTransformer


from seinfeldbot import SeinfeldChatbot

warnings.filterwarnings(action='ignore')

print('Model Loading...')

# Initializing IMDB
print('Initializing IMDB')
ia = imdb.IMDb()
series = ia.get_movie('0098904')
ia.update(series, 'episodes')
sorted(series['episodes'].keys())

# Loading episode IDs
with open('../data/episode_ids', 'rb') as f:
    episode_ids = pickle.load(f)
    print('Episode IDs loaded')

# Loading Sentence Transformer Model
roberta_url = 'https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/roberta-base-nli-stsb-mean-tokens.zip'
print('Loading Roberta Model...')
sentence_transformer_model = SentenceTransformer(roberta_url)
print('Roberta Model loaded')


# Opening SVC Model
with open('../data/svc_classification_model.pkl', 'rb') as f:
    svc_model = pickle.load(f)

if svc_model:
    print('Character Model Loaded')
else:
    print("Character Model Failed to Load")

# Loading Episode Vectors
with open('../data/roberta_episode_vectors.pkl', 'rb') as f:
    episode_vectors_dict = pickle.load(f) 

if episode_vectors_dict:
    print(f"Episode Vectors have loaded. The first episode (S01E01) size is {episode_vectors_dict.get('S01E01').shape}")
else:
    print('Vectors did not load')

bot = SeinfeldChatbot

bot.chat()