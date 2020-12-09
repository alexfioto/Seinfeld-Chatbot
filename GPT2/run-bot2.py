from aitextgen.TokenDataset import TokenDataset
from aitextgen.tokenizers import train_tokenizer
from aitextgen.utils import GPT2ConfigCPU
from aitextgen import aitextgen
import pickle
import spacy
import pandas as pd
import imdb
import warnings
import webbrowser

from seinfeldbot import SeinfeldChatbot

#from seinfeldbot import SeinfeldChatbot
warnings.filterwarnings(action='ignore')

print('Model Loading...')



# Opening SVC Model
with open('../data/svc_classification_model.pkl', 'rb') as f:
    svc_model = pickle.load(f)

if svc_model:
    print('Character Model Loaded')
else:
    print("Character Model Failed to Load")

with open('../data/light_episode_vectors.pkl', 'rb') as f:
    episode_vectors = pickle.load(f) 

if episode_vectors:
    print(f"Episode Vectors have loaded. The first episode (S01E01) size is {episode_vectors.get('S01E01').shape}")
else:
    print('Vectors did not load')

bot = SeinfeldChatbot

#bot.chat()