
# # GTP2 Generated Seinfeld ChatBot with Recommender System and Character Prediction


from aitextgen import aitextgen
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import pandas as pd
import imdb
import warnings
import webbrowser

# For Voice
import speech_recognition as sr
from gtts import gTTS
from pygame import mixer

from SeinfeldChatbot import SeinfeldChatbotLite

warnings.filterwarnings(action='ignore')


# Instantiate and start chat.
bot = SeinfeldChatbotLite()
bot.chat()


    

