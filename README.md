![](./data/Seinfeld_logo.png)


# Problem Statement:

Using dialogue from the television show Seinfeld, I am attempting to create a generative chatbot using accuracy as my objective performance metric. I will mainly utilize the keras deep learning library to implement my project.

I will be using accuracy as a metric in training my encoder NN, but ultimately success will be determined subjectively, i.e. how convincing is the chatbot?

The Seinfeld dialogue dataset has been downloaded from Kaggle. The data set consists of diaglogue from the complete series.

https://www.kaggle.com/thec03u5/seinfeld-chronicles

The model I will use is a seq2seq, otherwise known as an encoder-decoder, model.

Risks and assumptions of data: The model that I am building uses a lot of RAM and computing power. Some strings of dialogue are too long to be useful and have to be discarded. This can negatively effect my model as the dialogue will no longer be a true back and forth conversation.



# Background:

In mid 2015, two big events occured in my life: 1) I became a father, 2) I injured my lumbar spine in a sporting accident that made it near impossible to walk for 2 months. Needless to say, this was a very trying time for my family. Upset that I couldn't contribute more to normal daily tasks, I took it upon myself to rock my son to sleep for every nap while he lay on my chest. We called these "Daddy Naps". As I was immobilized by child and spine, I had a lot of time for Netflix and Hulu. The first show I watched all of the way through was Seinfeld. I had never watched it before, but figured it might give me some good dad joke material for down the road. Thankfully, this tough moment passed and I became much stronger for it. The bond created between myself and my son is invaluable and the rest gave me the time to heal and research a game-plan to come back stronger.  

Fast-forward five years. I find myself in starting an exciting and challenging new career! I figure, why not revisit this familiar medium to experiment with NLP and other data science techniques. 😁


# Executive Summary:

My goal with this project was to venture into the world of NLP and create a generative chatbot that is responsive, coherent and has functionality. I iterated through multiple variations of chatbots and have shared 3 in this notebook. 
I have divided up my executive summary into 3 sections corresponding to each iteration.


# Seq2Seq Model


**Summary:**
In preparation for the seq2seq model, the dataset was cleaned and missing values were removed. The dialogue included stage directions. The stage directions were removed. There were a few lines of dialogue that were very long. To make the model a bit more managable, the maximum length of dialogue was limited to 50 characters. The dialogue was then formatted properly to be accepted into the network. 




### Issues and concerns

**Size:**
When trained on all of the data, the model is very large. The filesize has been an issue when reading in to notebooks and python scripts. 

**Model Loading:**
The hidden and cell states are somehow altered when reading in a model from keras load_model(). I trained a few models in Google Colaboratory and saved them to my personal machine. The model did not perform as well after reloading in the model.

**Limiting Dialogue Length:**
Limiting the length of dialogue can impact the model negatively. The idea is that every line of dialogue is conversational. Cutting it short by length might make the dialogue disjointed and not perform well. Finding the correct type of data for this model is harder than actually implementing it!




### Moving forward:

**Different Dataset:** I would like to try this same model with a different data set. Short, conversational data would be the most appropriate. Online chat data would be great.

### Conclusion:
Overall, the Seq2Seq model concept is fascinating! I do believe there are better models architectures suited for this particular chatbot. I am eager to explore Chatterbot and GTP2 text generation. 



# Chatterbot

Chatterbot (https://chatterbot.readthedocs.io/en/stable/) module provides a simple and easy way to train a chatbot. This iteration is not venture to the sequence character predictions, however. The training dialougue is in the same format as the Seq2Seq model, however there is no deep learning involved. For this model, I used the "BestMatch" logic adapter using Jaccard Distance as the metric. 

The bot is trained and responses are stored in a SQLite database locally for quick calculations and extraction. As you interact with the bot, your responses are saved and theoretically will improve the bot's performance.

The simplicity of this model enabled me to deploy the bot on Flask and Streamlit locally as well as an online Heroku application. https://seinfeld-chatterbot.herokuapp.com/

### Issues and concerns:
**Not Generative by character:**
Although this bot works well, it is not generative by character. It is generative in the sense that it generates the most probabilistic response, but it is not as "artistic" as other models. 



### Moving Forward:
**Functionality:**
I may want to include additional functionality to this bot as with the GTP2 iteration.

**Select Different Context**
This type of chatbot would be very useful when it comes to certain types of utilities i.e. Kitchen Help Bot, Customer Service Bot, etc. I feel this architecture would work well with a partial rule-based chatbot. I may attempt to create a new chatbot with a different context in mind.


### Conclusions:
This is an easy to use module and very quick to train and store. The "Artistic" generativity leaves me wanting more, however I can see many useful applications of this style of chatbot


# GPT2

After playing around with GPT2 (Generative Pretrained Transformer-2) in a few online notebooks, I decided to create a chatbot with a slight variation for interface. This iteration of chatbot doesn't chat 1:1 but rather places you in the middle of dialogue. The model generates multiple lines of text which is then parsed down to coherent chunks. I included more functionality to this iteration than the two previous bots. 

I utilized Google Colab for initial training. You can find the notebook here: https://colab.research.google.com/drive/1WeL65rcgGCFK-8k5prHZg06i2w-Q_YLS. The aitextgen module is intended to work with Google and utilizes Google Drive for accessing text files and saving large models. Thanks to Max Woolf for the module and interface.

You can go here for more information on GTP2: https://github.com/openai/gpt-2



### spaCy Recommender

Utilizing spaCy's built in word embeddings, this chatbot comes loaded with word embeddings for each Seinfeld episode's dialogue. The bot stores user chat information, vectorizes, creates embeddings and calculates the cosine similarity between user_chat and every episode. 

The recommender utilizes the imdbpy api to grab episode names and plot. The recommender will also return a customized youtube search link for the episode of choice. 

### Character Classification

Using scikit-learn, I experiemented with multiple classification models to see how accurately I could classify dialogue from each of the main Seinfeld characters: Jerry, George, Elaine and Kramer.

Jerry has the most lines of any other character at 37%. My null model predicts 'Jerry' everytime. My best model was an Support Vector Machine with an accuracy of 43%. The dialogue was preprocessed using lemmatization and the text was vectorized using a TfidfVectorizer using english stop words. 

While the accuracy is not that much better than the null model, I dug into it a bit more and realized that a lot of the dialogue is very short with not much context. When input iconic quotes from each of the characters, the model performs significantly better. NOTE: The Jerry quote is made up.

```
jerry_quote = 'What is the deal with peanuts? Are they peas or are they nuts? Make up your mind!'
kramer_quote = 'I\'m out there, Jerry, and I\'m loving every minute of it!'
george_quote = 'I’m Disturbed, I’m Depressed, I’m Inadequate – I’ve Got It All!'
elaine_quote = "Maybe the dingo ate your baby!"

The model classifies the Jerry quote as Jerry.
The model classifies the Kramer quote as Kramer.
The model classifies the George quote as George.
The model classifies the Elaine quote as Elaine.
```

I gathered a sampling of iconic quotes from each of the characters. This mini-test resulted in a 14% jump in accuracy (57%). The model seems to misclassify Kramer and Elaine quotes as Jerry. The model does very well at distinguishing Jerry from George.  

I also performed a sentiment analysis for each of the characters. I used sentiment scores in a classification model this information in the 

### Issues and concerns:

The spaCy NLP file is too big. There must be a better way to summarize the text

### Moving Forward:

**GUI interface**:
First, I plan to implement a GUI interface for ease of use and better aesthetics. I have looked into a few options: Streamlit, Flask, and Telegram. I will be experimenting with them in the future. 

**Topic modeling to help with the recommender:**
Currently, the episode vectors have been created with spaCy's utilizing BERT uncased word embeddings. These NLP objects are very large and I am not sure how much signal there is. Recommenders are subjective and the only real way to know if working properly is to test in the field. Unfortunately, since this chatbot's backend files are so big, deployment might not be a feasible option at this stage.

In the notebook labeled "Topic-Modeling", I am exploring options to simply the recommender system using Gensim's keywords and summarization as well as gathering keywords from each episode based on BERT word embeddings. The latter option may be the best of both worlds. 

**Recommender Feedback**:
As stated before, there is no objective way to know if the recommender is performing effectivley. I eventually plan to design a feedback system that will rank user satisfaction with episode choices. 



### Conclusions:
Overall I hope to reduce the footprint of the model and eventually deploy to the public. The control flow of the model is working well but the aesthetics need work. 











