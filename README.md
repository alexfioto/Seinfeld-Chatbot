# Problem Statement:

Using dialogue from the television show Seinfeld, I am attempting to create a generative chatbot using a seq2seq model and accuracy as my objective performance metric. I will utilize the keras deep learning library to implement my project.

The Seinfeld dialogue dataset has been downloaded from Kaggle. The data set consists of diaglogue from the complete series.

https://www.kaggle.com/thec03u5/seinfeld-chronicles

The model I will use is a seq2seq, otherwise known as an encoder-decoder, model.

Risks and assumptions of data: The model that I am building uses a lot of RAM and computing power. Some strings of dialogue are too long to be useful and have to be discarded. This can negatively effect my model as the dialogue will no longer be a true back and forth conversation.

I will be using accuracy as a metric in training my encoder NN, but ultimately success will be determined subjectively, i.e. how convincing is the chatbot?

Please see my notebook, capstone-preprocess, for preliminary EDA and initial modeling. 

# Background:

In mid 2015, two big events occured in my life: 1) I became a father, 2) I injured my lumbar spine in a sporting accident that made it near impossible to walk for 2 months. Needless to say, this was a very trying time for my family. Upset that I couldn't contribute more to normal daily tasks, I took it upon myself to rock my son to sleep for every nap while he lay on my chest. We called these "Daddy Naps". As I was immobilized by child and spine, I had a lot of time for Netflix and Hulu. The first show I watched all of the way through was Seinfeld. I had never watched it before, but figured it might give me some good dad joke material for down the road. Thankfully, this tough moment passed and I became much stronger for it. The bond created between myself and my son is invaluable and the rest gave me the time to heal and research a game-plan to come back stronger.  

Fast-forward five years. I find myself in another tough moment: Losing my dream job due to COVID-19. I used Seinfeld as a medium while I heal up and bonded with my new born son. I figure, why not use it again to experiment with NLP and other data science techniques. This moment, too, shall pass and I will come out stronger. 


# Executive Summary:

My goal with this project was to venture into the world of NLP and create a chatbot that responsive, coherent and has functionality. I iterated through multiple variations of chatbots and have shared 3 in this notebook. Iteration 3: Seinfeld GPT-2 turned out to be the 'best', by my definition.

I have divided up my executive summary into 3 sections corresponding to each iteration.


## Seq2Seq Model

In preparation for the seq2seq model, the dataset was cleaned and missing values were removed. The dialogue included stage directions. The stage directions were removed. There were a few lines of dialogue that were very long. To make the model a bit more managable, the maximum length of dialogue was limited to 50 characters.

The dialogue was then formatted properly to be accepted into the network. 





