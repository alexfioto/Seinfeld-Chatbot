from aitextgen.TokenDataset import TokenDataset
from aitextgen.tokenizers import train_tokenizer
from aitextgen.utils import GPT2ConfigCPU
from aitextgen import aitextgen
import pickle
import spacy
import pandas as pd
import imdb
import warnings



class SeinfeldChatbot():
    def __init__(self, name='Buddy', transformer='en_trf_bertbaseuncased_lg', fp='/Users/alexander.fioto/Models/Labeled-Seinfeld-Model/', temperature = .4):
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
        
        
    def change_temp(self, temp):
        if temp < 0 or temp > 1:
            raise ValueError('Value must be greater than 0 and less than or equal to 1')
        else:
            self.temperature = temp
        
        
    def greet(self):
        res = input("Welcome to the SeinfeldChatbot! Do you want to chat?")
        if res.lower() not in self.negative_responses:
            res = input("GEORGE: Can I get your name, pal?")
            if res in self.negative_responses:
                print('GEORGE: Fine. Stay anonymous. See if I care.')
            else:
                self.name = res
                self.user_name_title = self.name.upper() + ': '
            self.greeted = True
            self.chat()
        else:
            print('JERRY: OK. Fine. Leave. GO!')
    
    def generate_response(self, text_input):
        if text_input[-1] not in self.punctuation:
            text_input += '.'
        self.chat_dialogue += ' ' + text_input
        text = self.model.generate(prompt = f'{self.user_name_title}' + text_input,
                                            temperature = .4,
                                            return_as_list = True)
        self.split_text_ = text[0].split('\n\n')
        if len(self.split_text_[-1]) < 16:
            self.split_text_.pop()
        final_text = ''.join(self.split_text_)
        self.chat_dialogue_complete += final_text + '\n'
        print(final_text)
        
    
    def chat(self):
        if self.greeted == True:
            print(f'JERRY: What do you want, {self.name}?')
            chat = True
            while chat:
                text_input = input(f'{self.user_name_title}')
                if text_input in self.exit_commands:
                    chat = False
                    print('KRAMER: Who turns down a junior mint?')
                    print('Thanks for chatting!')
                elif 'recommend' in text_input:
                    res = input("JERRY: Did you want an episode recommendation? It might take a minute.")
                    if res.lower() in self.positive_responses:
                        if self.recommender_initialized:
                            self.episode_recommendation()
                        else:
                            self.initialize_recommender()
                    else:
                        print('JERRY: Oh. well what you said wasn\'t clear.')
                        
                
                elif 'sound like' in text_input.lower():
                        self.predict_character()

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
            res = input('You need to initialize the recommender system. Would you like to initialize?')
            if res.lower() in self.positive_responses:
                self.initialize_recommender()
            else:
                print('OK. Hope you come back later!')
        
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
                print('JERRY: Thanks for you patience. That took way too long.')
            else:
                print('KRAMER: It looks like you haven\'t chatted yet. Please chat for a while and come back!')
    
    def episode_recommendation(self):
        
        
        if not self.recommender_initialized:
            res = input('JERRY: You need to initialize the recommender. Want to do it?')
            if res.lower() in self.positive_responses:
                self.initialize_recommender()
            else:
                print('JERRY: OK.')
        
        
        elif not self.similarity_scores:
            res = input('ELAINE: You need to get similarity scores first. Want to do grab them?')
            if res.lower() in self.positive_responses:
                self.update_similarities()
            else:
                print('GEORGE: Fine. Have it that way.')
    
            
        else:
            for i in range(len(self.scores_list_)):
                episode = self.series_['episodes'][self.scores_list_[i][0]][self.scores_list_[i][1]]
                title = episode['title']
                plot = episode['plot']
                res = input(f'JERRY: Based on your chat dialogue, I would recommend you check out Seinfeld Season {self.scores_list_[i][0]}, episode {self.scores_list_[i][1]}, "{title}". Do you want to know the plot?')
                if res.lower() in self.positive_responses:
                    print(plot)
                res = input('Do you want a link to the show?')
                if res.lower() in self.positive_responses:
                    print(f'https://www.youtube.com/results?search_query=seinfeld+season+{self.scores_list_[i][0]}+episode+{self.scores_list_[i][1]}')
                res = input('JERRY: Do you want another recommendation?')
                if res == 'no':
                    print('JERRY: OK.')
                    break

            
    ##### Character Predictor#####
    
    def predict_character(self, text=None):
        if text:
            prediction = svc_model.predict([text])
        else:
            prediction = svc_model.predict([self.chat_dialogue])
            
        if prediction == 0:
            print('You sound like Jerry!')
        elif prediction == 1:
            print('You sound like George!')
        elif prediction == 1:
            print('You sound like Kramer!')
        else:
            print('You sound like Elaine!'