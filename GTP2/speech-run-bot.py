
# # GTP2 Generated Seinfeld ChatBot with Recommender System and Character Prediction


from aitextgen.TokenDataset import TokenDataset
from aitextgen.tokenizers import train_tokenizer
from aitextgen.utils import GPT2ConfigCPU
from aitextgen import aitextgen
import pickle
import spacy
import pandas as pd
import imdb
import warnings
import time
import webbrowser

import speech_recognition as sr
from gtts import gTTS
from pygame import mixer

#from seinfeldbot import SeinfeldChatbot


warnings.filterwarnings(action='ignore')

print('Model Loading...')

# Loading dictionary with all of the episode names and dialogues
with open('../data/episode_dialogues.pkl', 'rb') as f:
    episode_dialogues = pickle.load(f)


# Loading BERT embeddings for Seinfeld Epiosdes
with open('/Users/alexander.fioto/Models/seinfeld_bert_spacy.pkl', 'rb') as f:
    seinfeld_vectors = pickle.load(f)


# Opening SVC Model
with open('../data/svc_classification_model.pkl', 'rb') as f:
    svc_model = pickle.load(f)

if svc_model:
    print('Character Model Loaded')
else:
    print("Character Model Failed to Load")


class SeinfeldChatbot():
    def __init__(self, name='Buddy', transformer='en_trf_bertbaseuncased_lg', fp='/Users/alexander.fioto/Models/Larger-Seinfeld-Model/', temperature = .4):
        self.user_name_title = 'USER: '
        self.chat_dialogue = ''
        self.chat_dialogue_complete = ''
        self.name = name
        self.fp = fp
        self.greeted = False
        self.similarity_scores = None
        self.temperature = temperature
        self.exit_commands = ['bye', 'exit', 'i have to go', 'later', 'gtg', 'stop', 'end', 'done']
        self.positive_responses = ['yes', 'yep', 'sure', 'definitely', 'y']
        self.negative_responses = ['no', 'no thanks', 'nope', 'nah', 'n']
        self.punctuation = ['.', '!', '?']
        self.model = aitextgen(model= fp + "pytorch_model.bin",
                               config = fp + 'config.json', 
                               vocab_file=fp + 'aitextgen-vocab.json',
                               merges_file=fp + 'aitextgen-merges.txt')
        self.transformer = transformer
        self.recommender_initialized = False
        print("Model Loaded!")
    
    def bot_speak(self, text):
        speech = gTTS(text=text, lang='en', slow=False)
        speech.save('text.mp3')
        mixer.init()
        mixer.music.load('text.mp3')
        mixer.music.play()
        print(text)
        time.sleep(3)

    def speech_input(self):
        r = sr.Recognizer()
        time.sleep(3)
        with sr.Microphone() as source:
            print("Say something!")
            audio = r.listen(source)
        text = r.recognize_google(audio)
        return text
        
        
    def change_temp(self, temp):
        if temp < 0 or temp > 1:
            raise ValueError('Value must be greater than 0 and less than or equal to 1')
        else:
            self.temperature = temp
        
        
    def greet(self):
        
        self.bot_speak("Welcome to the SeinfeldChatbot! Do you want to chat?")
        res = self.speech_input()
        if res.lower() not in self.negative_responses:
            self.bot_speak("Can I get your name?")
            res = self.speech_input()
            if res in self.negative_responses:
                self.bot_speak('Fine. Stay anonymous.')
            else:
                self.name = res
                self.user_name_title = self.name.upper() + ': '
            self.greeted = True
            self.chat()
        else:
            self.bot_speak('Come back when you are feeling more loquacious')
    
    def generate_response(self, text_input):
        if text_input[-1] not in self.punctuation:
            text_input += '.'
        self.chat_dialogue += ' ' + text_input
        text = self.model.generate(prompt = f'{self.user_name_title}' + text_input,
                                            temperature = .4,
                                            return_as_list = True)
        self.split_text_ = text[0].split('\n\n')
        self.split_text_.pop()
        final_text = ''.join(self.split_text_[:3])
        self.chat_dialogue_complete += final_text + '\n'
        self.bot_speak(final_text)
        time.sleep(8)
        
    
    def chat(self):
        if self.greeted == True:
            self.bot_speak(f'JERRY: What do you want, {self.name}?')
            chat = True
            while chat:

                text_input = self.speech_input()

                if text_input in self.exit_commands:
                    chat = False
                    self.bot_speak('KRAMER: Who turns down a junior mint?')
                    self.bot_speak('Thanks for chatting!')

                elif 'recommend' in text_input:
                    self.bot_speak("Did you want an episode recommendation? It might take a minute.")
                    res = self.speech_input()
                    if res.lower() in self.positive_responses:
                        if self.recommender_initialized:
                            self.episode_recommendation()
                        else:
                            self.initialize_recommender()
                    else:
                        self.bot_speak('I didn\'t understand.')
                
                elif 'sound like' in text_input.lower():
                        self.predict_character()
                
                elif 'temp' in text_input.lower():
                    self.bot_speak('Did you want to change the temperature (or creativity)?')
                    res = self.speech_input()
                    if res.lower() in self.positive_responses:
                        res = input('JERRY: Type a float greater than 0 and less than or equal to 1')
                        self.change_temp(float(res))

                else:
                    self.generate_response(text_input)
        else:
            self.greet()

    
    
    #### Episode Recommender #### 
    def initialize_recommender(self):
        '''
        This method initializes the recommender engine. You may use other pretrained transformers such as
        en_trf_bertbaseuncased_lg. If you use BERT, you will see more accurate results but it will take longer
        to load!
        '''
        if not self.recommender_initialized:
            self.nlp_ = spacy.load(self.transformer)
            self.recommender_initialized = True
            df = pd.read_csv('../data/clean_scripts.csv', index_col=0)
            self.episodes_ = df['SEID'].unique()

            ia = imdb.IMDb()
            self.series_ = ia.get_movie('0098904')
            ia.update(self.series_, 'episodes')
            sorted(self.series_['episodes'].keys())
            print('Recommender Initialized!')
            self.update_similarities()
            self.episode_recommendation()
        else:
            print("The recommender is already initialized.")
        
        

    def update_similarities(self):
        if not self.recommender_initialized:
            self.bot_speak('You need to initialize the recommender system. Would you like to initialize?')
            res = self.speech_input()
            if res.lower() in self.positive_responses:
                self.initialize_recommender()
            else:
                self.bot_speak('OK. Hope you come back later!')
        
        else:

            if self.chat_dialogue:
                similarity_scores = []
                for episode in self.episodes_:
                    doc1 = self.nlp_(self.chat_dialogue)
                    doc2 = seinfeld_vectors[episode]
                    similarity_scores.append((episode, doc1.similarity(doc2)))
                similarity_scores.sort(key=lambda x: x[1], reverse = True)
                self.similarity_scores = similarity_scores
                self.scores_list_ = []
                for i in range(len(self.similarity_scores)):
                    self.scores_list_.append([int(self.similarity_scores[i][0][1:3]), int(self.similarity_scores[i][0][-2:])])
                self.bot_speak('Thanks for you patience. That took way too long.')
                time.sleep(5)
            else:
                self.bot_speak('It looks like you haven\'t chatted yet. Please chat for a while and come back!')
    
    def episode_recommendation(self):
        
        
        if not self.recommender_initialized:
            self.bot_speak('You need to initialize the recommender. Want to do it?')
            res = self.speech_input()
            if res.lower() in self.positive_responses:
                self.initialize_recommender()
            else:
                self.bot_speak('OK.')
        
        
        elif not self.similarity_scores:
            self.bot_speak('You need to get similarity scores first. Want to do grab them?')
            res = self.speech_input()
            if res.lower() in self.positive_responses:
                self.update_similarities()
            else:
                self.bot_speak('Fine. Have it that way.')
    
            
        else:
            for i in range(len(self.scores_list_)):
                try:

                    episode = self.series_['episodes'][self.scores_list_[i][0]][self.scores_list_[i][1]]
                    title = episode['title']
                    plot = episode['plot']
                    self.bot_speak(f'Based on your chat dialogue, I would recommend you check out Seinfeld Season {self.scores_list_[i][0]}, episode {self.scores_list_[i][1]}, "{title}". Do you want to know the plot?')
                    res = self.speech_input()
                    if res.lower() in self.positive_responses:
                        print(plot)
                    self.bot_speak('Do you want to watch it?')
                    res = self.speech_input()
                    if res.lower() in self.positive_responses:
                        webbrowser.open_new(f'https://youtube.com/results?search_query=seinfeld+season+{self.scores_list_[i][0]}+episode+{self.scores_list_[i][1]}')
                    self.bot_speak('Do you want another recommendation?')
                    res = self.speech_input()
                    if res.lower not in self.positive_responses:
                        self.bot_speak('OK.')
                        break
                except:
                    continue
            
    ##### Character Predictor#####
    
    def predict_character(self, text=None):
        if text:
            prediction = svc_model.predict([text])
        else:
            prediction = svc_model.predict([self.chat_dialogue])
            
        if prediction == 0:
            self.bot_speak('You sound like Jerry!')
        elif prediction == 1:
            self.bot_speak('You sound like George!')
        elif prediction == 1:
            self.bot_speak('You sound like Kramer!')
        else:
            self.bot_speak('You sound like Elaine!')
   
    
        
        


# Instantiate and start chat.
bot = SeinfeldChatbot()
bot.chat()

