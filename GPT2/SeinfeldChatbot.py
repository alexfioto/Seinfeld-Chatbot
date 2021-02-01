from aitextgen import aitextgen
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import pandas as pd
import imdb
import warnings
import webbrowser

import speech_recognition as sr
from gtts import gTTS
from pygame import mixer









class SeinfeldChatbotLite():
    def __init__(self, name='Pal', fp='/Users/alexander.fioto/Models/Larger-Seinfeld-Model/', temperature = .4):
        print('Loading Seinfeld Chatbot Lite...')
        self.user_name_title = 'USER: '
        self.chat_dialogue = ''
        self.last_line = ''
        self.chat_dialogue_complete = ''
        self.name = name
        self.fp = fp
        self.greeted = False
        self.temperature = temperature
        self.exit_commands = ['bye', 'exit', 'i have to go', 'later', 'gtg', 'stop', 'end', 'done']
        self.positive_responses = ['yes', 'yep', 'sure', 'definitely', 'y']
        self.negative_responses = ['no', 'no thanks', 'nope', 'nah', 'n']
        self.punctuation = ['.', '!', '?']
        self.model = aitextgen(model= fp + "pytorch_model.bin",
                               config = fp + 'config.json', 
                               vocab_file=fp + 'aitextgen-vocab.json',
                               merges_file=fp + 'aitextgen-merges.txt')
        self.recommender_initialized = False

        with open('../data/distilbert_episode_vectors.pkl', 'rb') as f:
            self.seinfeld_episode_vectors = pickle.load(f)
        with open('../data/svc_classification_model.pkl', 'rb') as f:
            self.character_model = pickle.load(f)
    
        print("...Seinfeld Chatbot Lite Loaded!")
        
        
    def change_temp(self, temp):
        if temp < 0 or temp > 1:
            raise ValueError('Value must be greater than 0 and less than or equal to 1')
        else:
            self.temperature = temp
        
        
    def greet(self):
        res = input("Welcome to the SeinfeldChatbot! Do you want to chat?\n>>> ")
        if res.lower() not in self.negative_responses:
            res = input("GEORGE: Can I get your name, pal?\n>>> ")
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
        self.last_line = text_input
        self.chat_dialogue += ' ' + text_input
        text = self.model.generate(prompt= f'{self.user_name_title}' + text_input,
                                            temperature = self.temperature,
                                            return_as_list = True)
        self.split_text_ = text[0].split('\n\n')
        self.split_text_.pop()
        final_text = ''.join(self.split_text_[:3])
        self.chat_dialogue_complete += final_text + '\n'
        print(final_text)
        
    
    def chat(self):
        if self.greeted == True:
            print(f'JERRY: What do you want, {self.name}?')
            chat = True
            while chat:

                text_input = input('>>> ')

                if text_input in self.exit_commands:
                    chat = False
                    print('KRAMER: Who turns down a junior mint?')
                    print('Thanks for chatting!')

                elif 'recommend' in text_input:
                    res = input("JERRY: Did you want an episode recommendation? It might take a minute.\n>>> ")
                    if res.lower() in self.positive_responses:
                        if self.recommender_initialized:
                            self.episode_recommendation()
                        else:
                            self.initialize_recommender()
                    else:
                        print('JERRY: Oh. well what you said wasn\'t clear.')
                
                elif 'sound like' in text_input.lower():
                        self.predict_character()
                
                elif 'temp' in text_input.lower():
                    res = input('JERRY: Did you want to change the temperature (or creativity)?\n>>> ')
                    if res.lower() in self.positive_responses:
                        res = input('JERRY: Input a float greater than 0 and less than or equal to 1\n>>> ')
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
            ia = imdb.IMDb()
            self.series_ = ia.get_movie('0098904')
            ia.update(self.series_, 'episodes')
            sorted(self.series_['episodes'].keys())
            print('Recommender Initialized!')
            self.recommender_initialized = True
            self.update_similarities()
            self.episode_recommendation()
        else:
            print("The recommender is already initialized.")
        
    
    def get_similarities(self, model ='stsb-distilbert-base'):
        similarity_list = []
        ordered_episodes = []

        model = SentenceTransformer(model)
        dialogue_vector = model.encode(self.chat_dialogue).reshape(1,-1)
        for episode, vector in self.seinfeld_episode_vectors.items():
            similarity_list.append((episode, cosine_similarity(dialogue_vector, vector)[0][0]))
        similarity_list.sort(key=lambda x: x[1], reverse=True)
        ordered_episodes = []
        for i in range(len(similarity_list)):
            ordered_episodes.append([int(similarity_list[i][0][1:3]), int(similarity_list[i][0][-2:])])
        return similarity_list, ordered_episodes


    def update_similarities(self):
        if not self.recommender_initialized:
            res = input('You need to initialize the recommender system. Would you like to initialize?\n>>> ')
            if res.lower() in self.positive_responses:
                self.initialize_recommender()
            else:
                print('OK. Hope you come back later!')
        
        else:
            if self.chat_dialogue:
                self.similarity_scores_, self.ordered_episodes_ = self.get_similarities()
                print('JERRY: Thanks for you patience. That took way too long.')
            else:
                print('KRAMER: It looks like you haven\'t chatted yet. Please chat for a while and come back!')
    
    def episode_recommendation(self):
        
        
        if not self.recommender_initialized:
            res = input('JERRY: You need to initialize the recommender. Want to do it?\n>>> ')
            if res.lower() in self.positive_responses:
                self.initialize_recommender()
            else:
                print('JERRY: OK.')
        
        
        elif not self.similarity_scores_:
            res = input('ELAINE: You need to get similarity scores first. Want to do grab them?\n>>> ')
            if res.lower() in self.positive_responses:
                self.update_similarities()
            else:
                print('GEORGE: Fine. Have it that way.')
    
            
        else:

            for i in range(len(self.similarity_scores_)):
                try:
                    season_num = self.ordered_episodes_[i][0]
                    episode_num = self.ordered_episodes_[i][1]
                    episode = self.series_['episodes'][season_num][episode_num]
                    title = episode['title']
                    plot = episode['plot']
                    res = input(f'JERRY: Based on your chat dialogue, I would recommend you check out Seinfeld Season {season_num}, episode {episode_num}, "{title}". Do you want to know the plot?\n>>> ')
                    if res.lower() in self.positive_responses:
                        print(plot)
                    res = input('Do you want to watch the show?\n>>> ')
                    if res.lower() in self.positive_responses:
                        webbrowser.open_new(f'https://youtube.com/results?search_query=seinfeld+{title}+season+{season_num}+episode+{episode_num}')
                    res = input('JERRY: Do you want another recommendation?\n>>> ')
                    if res == 'no':
                        print('JERRY: OK.')
                        break
                except:
                    continue
            
    ##### Character Predictor#####
    
    def predict_character(self, text=None):
        if text:
            prediction = self.character_model.predict([text])
        else:
            prediction = self.character_model.predict([self.last_line])
            
        if prediction == 0:
            print('You sound like Jerry!')
        elif prediction == 1:
            print('You sound like George!')
        elif prediction == 2:
            print('You sound like Kramer!')
        else:
            print('You sound like Elaine!')
   

























# class SeinfeldChatbotVoice():

#     def __init__(self, name='Buddy', fp='/Users/alexander.fioto/Models/Larger-Seinfeld-Model/', temperature = .4):
#         print('Loading Seinfeld Chatbot Voice...')
#         self.user_name_title = 'USER: '
#         self.chat_dialogue = ''
#         self.chat_dialogue_complete = ''
#         self.name = name
#         self.fp = fp
#         self.greeted = False
#         self.similarity_scores = None
#         self.temperature = temperature
#         self.exit_commands = ['bye', 'exit', 'i have to go', 'later', 'gtg', 'stop', 'end', 'done']
#         self.positive_responses = ['yes', 'yep', 'sure', 'definitely', 'y']
#         self.negative_responses = ['no', 'no thanks', 'nope', 'nah', 'n']
#         self.punctuation = ['.', '!', '?']
#         self.model = aitextgen(model= fp + "pytorch_model.bin",
#                                config = fp + 'config.json', 
#                                vocab_file=fp + 'aitextgen-vocab.json',
#                                merges_file=fp + 'aitextgen-merges.txt')
#         with open('../data/distilbert_episode_vectors.pkl', 'rb') as f:
#             self.seinfeld_episode_vectors = pickle.load(f)
#         with open('../data/svc_classification_model.pkl', 'rb') as f:
#             self.character_model = pickle.load(f)
#         self.recommender_initialized = False
#         print("...Seinfeld Chatbot Voice Loaded!")
    
#     def bot_speak(self, text):
#         speech = gTTS(text=text, slow=False)
#         speech.save('text.mp3')
#         mixer.init()
#         mixer.music.load('text.mp3')
#         mixer.music.play()
#         print(text)
#         time.sleep(2)

#     def speech_input(self):
#         r = sr.Recognizer()
#         time.sleep(3)
#         with sr.Microphone() as source:
#             print("Say something!")
#             audio = r.listen(source)
#         text = r.recognize_google(audio)
#         if text in self.exit_commands:
#             self.exit()
#         return text

#     def exit(self):
#         self.bot_speak('Thanks for Chatting!')
#         exit()
        
        
#     def change_temp(self, temp):
#         if temp < 0 or temp > 1:
#             raise ValueError('Value must be greater than 0 and less than or equal to 1')
#         else:
#             self.temperature = temp
        
        
#     def greet(self):
        
#         self.bot_speak("Welcome to the SeinfeldChatbot! Do you want to chat?")
#         res = self.speech_input()
#         if res.lower() not in self.negative_responses:
#             self.bot_speak("Can I get your name?")
#             res = self.speech_input()
#             if res in self.negative_responses:
#                 self.bot_speak('Fine. Stay anonymous.')
#             else:
#                 self.name = res
#                 self.user_name_title = self.name.upper() + ': '
#             self.greeted = True
#             self.chat()
#         else:
#             self.bot_speak('Come back when you are feeling more loquacious')
    
#     def generate_response(self, text_input):
#         if text_input[-1] not in self.punctuation:
#             text_input += '.'
#         self.chat_dialogue += ' ' + text_input
#         text = self.model.generate(prompt = f'{self.user_name_title}' + text_input,
#                                             temperature = .4,
#                                             return_as_list = True)
#         self.split_text_ = text[0].split('\n\n')
#         self.split_text_.pop()
#         final_text = ''.join(self.split_text_[:3])
#         self.chat_dialogue_complete += final_text + '\n'
#         self.bot_speak(final_text)
#         time.sleep(8)
        
    
#     def chat(self):
#         if self.greeted == True:
#             self.bot_speak(f'JERRY: What do you want, {self.name}?')
#             chat = True
#             while chat:

#                 text_input = self.speech_input()

#                 if text_input in self.exit_commands:
#                     chat = False
#                     self.bot_speak('KRAMER: Who turns down a junior mint?')
#                     self.bot_speak('Thanks for chatting!')

#                 elif 'recommend' in text_input:
#                     self.bot_speak("Did you want an episode recommendation? It might take a minute.")
#                     res = self.speech_input()
#                     if res.lower() in self.positive_responses:
#                         if self.recommender_initialized:
#                             self.episode_recommendation()
#                         else:
#                             self.initialize_recommender()
#                     else:
#                         self.bot_speak('I didn\'t understand.')
                
#                 elif 'sound like' in text_input.lower():
#                         self.predict_character()
                
#                 elif 'temp' in text_input.lower():
#                     self.bot_speak('Did you want to change the temperature (or creativity)?')
#                     res = self.speech_input()
#                     if res.lower() in self.positive_responses:
#                         res = input('JERRY: Type a float greater than 0 and less than or equal to 1')
#                         self.change_temp(float(res))

#                 else:
#                     self.generate_response(text_input)
#         else:
#             self.greet()

    
    
#     #### Episode Recommender #### 
#     def initialize_recommender(self):
#         '''
#         This method initializes the recommender engine. You may use other pretrained transformers such as
#         en_trf_bertbaseuncased_lg. If you use BERT, you will see more accurate results but it will take longer
#         to load!
#         '''
#         if not self.recommender_initialized:
#             self.nlp_ = spacy.load(self.transformer)
#             self.recommender_initialized = True
#             df = pd.read_csv('../data/clean_scripts.csv', index_col=0)
#             self.episodes_ = df['SEID'].unique()

#             ia = imdb.IMDb()
#             self.series_ = ia.get_movie('0098904')
#             ia.update(self.series_, 'episodes')
#             sorted(self.series_['episodes'].keys())
#             print('Recommender Initialized!')
#             self.update_similarities()
#             self.episode_recommendation()
#         else:
#             print("The recommender is already initialized.")
        
        

#     def update_similarities(self):
#         if not self.recommender_initialized:
#             self.bot_speak('You need to initialize the recommender system. Would you like to initialize?')
#             res = self.speech_input()
#             if res.lower() in self.positive_responses:
#                 self.initialize_recommender()
#             else:
#                 self.bot_speak('OK. Hope you come back later!')
        
#         else:

#             if self.chat_dialogue:
#                 similarity_scores = []
#                 for episode in self.episodes_:
#                     doc1 = self.nlp_(self.chat_dialogue)
#                     doc2 = seinfeld_vectors[episode]
#                     similarity_scores.append((episode, doc1.similarity(doc2)))
#                 similarity_scores.sort(key=lambda x: x[1], reverse = True)
#                 self.similarity_scores = similarity_scores
#                 self.scores_list_ = []
#                 for i in range(len(self.similarity_scores)):
#                     self.scores_list_.append([int(self.similarity_scores[i][0][1:3]), int(self.similarity_scores[i][0][-2:])])
#                 self.bot_speak('Thanks for your patience.')
#                 time.sleep(3)
#             else:
#                 self.bot_speak('It looks like you haven\'t chatted yet. Please chat for a while and come back!')
    
#     def episode_recommendation(self):
        
        
#         if not self.recommender_initialized:
#             self.bot_speak('You need to initialize the recommender. Want to do it?')
#             res = self.speech_input()
#             if res.lower() in self.positive_responses:
#                 self.initialize_recommender()
#             else:
#                 self.bot_speak('OK.')
        
        
#         elif not self.similarity_scores:
#             self.bot_speak('You need to get similarity scores first. Want to do grab them?')
#             res = self.speech_input()
#             if res.lower() in self.positive_responses:
#                 self.update_similarities()
#             else:
#                 self.bot_speak('Fine. Have it that way.')
    
            
#         else:
#             for i in range(len(self.scores_list_)):
#                 try:

#                     episode = self.series_['episodes'][self.scores_list_[i][0]][self.scores_list_[i][1]]
#                     title = episode['title']
#                     plot = episode['plot']
#                     self.bot_speak(f'Based on your chat dialogue, I would recommend you check out Seinfeld Season {self.scores_list_[i][0]}, episode {self.scores_list_[i][1]}, "{title}". Do you want to know the plot?')
#                     time.sleep(6)
#                     res = self.speech_input()
#                     if res.lower() in self.positive_responses:
#                         self.bot_speak(plot)
#                         time.sleep(15)
#                     self.bot_speak('Do you want to watch it?')
#                     res = self.speech_input()
#                     if res.lower() in self.positive_responses:
#                         self.bot_speak('Great. I hope you enjoy!')
#                         webbrowser.open_new(f'https://youtube.com/results?search_query=seinfeld+season+{self.scores_list_[i][0]}+episode+{self.scores_list_[i][1]}')
#                         input('Enter any key to continue')
#                         break
#                     self.bot_speak('Do you want another recommendation?')
#                     res = self.speech_input()
#                     if res.lower not in self.positive_responses:
#                         self.bot_speak('OK.')
#                         break
#                 except:
#                     continue
            
#     ##### Character Predictor#####
    
#     def predict_character(self, text=None):
#         if text:
#             prediction = svc_model.predict([text])
#         else:
#             prediction = svc_model.predict([self.chat_dialogue])
            
#         if prediction == 0:
#             self.bot_speak('You sound like Jerry!')
#         elif prediction == 1:
#             self.bot_speak('You sound like George!')
#         elif prediction == 1:
#             self.bot_speak('You sound like Kramer!')
#         else:
#             self.bot_speak('You sound like Elaine!')
























# class SeinfeldChatbotClassic():
#     def __init__(self, name='Buddy', transformer='en_trf_bertbaseuncased_lg', fp='/Users/alexander.fioto/Models/Larger-Seinfeld-Model/', temperature = .4):
#         self.user_name_title = 'USER: '
#         self.chat_dialogue = ''
#         self.chat_dialogue_complete = ''
#         self.name = name
#         self.fp = fp
#         self.greeted = False
#         self.similarity_scores = None
#         self.temperature = temperature
#         self.exit_commands = ['bye', 'exit', 'i have to go', 'later', 'gtg', 'stop', 'end', 'done']
#         self.positive_responses = ['yes', 'yep', 'sure', 'definitely', 'y']
#         self.negative_responses = ['no', 'no thanks', 'nope', 'nah', 'n']
#         self.punctuation = ['.', '!', '?']
#         self.model = aitextgen(model= fp + "pytorch_model.bin",
#                                config = fp + 'config.json', 
#                                vocab_file=fp + 'aitextgen-vocab.json',
#                                merges_file=fp + 'aitextgen-merges.txt')
#         self.transformer = transformer
#         self.sentence_transformer_model = sentence_transformer_model
#         self.episode_vectors_dict = episode_vectors_dict
#         self.episode_ids = episode_ids

#         print("Model Loaded!")
        
        
#     def change_temp(self, temp):
#         if temp < 0 or temp > 1:
#             raise ValueError('Value must be greater than 0 and less than or equal to 1')
#         else:
#             self.temperature = temp
        
        
#     def greet(self):
#         res = input("Welcome to the SeinfeldChatbot! Do you want to chat?")
#         if res.lower() not in self.negative_responses:
#             res = input("GEORGE: Can I get your name, pal?")
#             if res in self.negative_responses:
#                 print('GEORGE: Fine. Stay anonymous. See if I care.')
#             else:
#                 self.name = res
#                 self.user_name_title = self.name.upper() + ': '
#             self.greeted = True
#             self.chat()
#         else:
#             print('JERRY: OK. Fine. Leave. GO!')
    
#     def generate_response(self, text_input):
#         if text_input[-1] not in self.punctuation:
#             text_input += '.'
#         self.chat_dialogue += ' ' + text_input
#         text = self.model.generate(prompt = f'{self.user_name_title}' + text_input,
#                                             temperature = self.temperature,
#                                             return_as_list = True)
#         self.split_text_ = text[0].split('\n\n')
#         if len(self.split_text_[-1]) < 16:
#             self.split_text_.pop()
#         final_text = ''.join(self.split_text_)
#         self.chat_dialogue_complete += final_text + '\n'
#         print(final_text)
        
    
#     def chat(self):
#         if self.greeted == True:
#             print(f'JERRY: What do you want, {self.name}?')
#             chat = True
#             while chat:
#                 text_input = input(f'{self.user_name_title}')
#                 if text_input in self.exit_commands:
#                     chat = False
#                     print('KRAMER: Who turns down a junior mint?')
#                     print('Thanks for chatting!')
#                 elif 'recommend' in text_input:
#                     res = input("JERRY: Did you want an episode recommendation? It might take a minute.")
#                     if res.lower() in self.positive_responses:
#                         if self.recommender_initialized:
#                             self.episode_recommendation()
#                         else:
#                             self.initialize_recommender()
#                     else:
#                         print('JERRY: Oh. well what you said wasn\'t clear.')
                        
                
#                 elif 'sound like' in text_input.lower():
#                         self.predict_character()

#                 else:
#                     self.generate_response(text_input)
#         else:
#             self.greet()

    
    
#     #### Episode Recommender #### 

#     def update_similarities(self):
#         if self.chat_dialogue:
#             similarity_scores = []
#             user_dialogue_vector = self.sentence_transformer_model.encode(self.chat_dialogue).reshape(1, -1)
#             for episode, vector in self.episode_vectors_dict.items():
#                 similarity_scores.append((episode, cosine_similarity(user_dialogue_vector, vector)[0][0])
#             similarity_scores.sort(key=lambda x: x[1], reverse = True)
#             self.similarity_scores = similarity_scores
#             self.scores_list_ = []
#             for i in range(len(self.similarity_scores)):
#                 self.scores_list_.append([int(self.similarity_scores[i][0][1:3]), int(self.similarity_scores[i][0][-2:])])
#             print('JERRY: Thanks for you patience. That took way too long.')
#         else:
#             print('KRAMER: It looks like you haven\'t chatted yet. Please chat for a while and come back!')

    
#     def episode_recommendation(self):
    
#         if not self.similarity_scores:
#             res = input('ELAINE: You need to get similarity scores first. Want to do grab them?')
#             if res.lower() in self.positive_responses:
#                 self.update_similarities()
#             else:
#                 print('GEORGE: Fine. Have it that way.')
    
            
#         else:
#             for i in range(len(self.scores_list_)):
#                 episode = self.series_['episodes'][self.scores_list_[i][0]][self.scores_list_[i][1]]
#                 title = episode['title']
#                 plot = episode['plot']
#                 res = input(f'JERRY: Based on your chat dialogue, I would recommend you check out Seinfeld Season {self.scores_list_[i][0]}, episode {self.scores_list_[i][1]}, "{title}". Do you want to know the plot?')
#                 if res.lower() in self.positive_responses:
#                     print(plot)
#                 res = input('Do you want to watch the show?')
#                 if res.lower() in self.positive_responses:
#                     webbrowser.open_new(f'https://youtube.com/results?search_query=seinfeld+season+{episode}')
#                 res = input('JERRY: Do you want another recommendation?')
#                 if res == 'no':
#                     print('JERRY: OK.')
#                     break

            
#     ##### Character Predictor#####
    
#     def predict_character(self, text=None):
#         if text:
#             prediction = svc_model.predict([text])
#         else:
#             prediction = svc_model.predict([self.chat_dialogue])
            
#         if prediction == 0:
#             print('You sound like Jerry!')
#         elif prediction == 1:
#             print('You sound like George!')
#         elif prediction == 1:
#             print('You sound like Kramer!')
#         else:
#             print('You sound like Elaine!')