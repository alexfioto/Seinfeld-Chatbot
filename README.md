# Problem Statement:

Using dialogue from the television show Seinfeld, I am attempting to create a generative chatbot using a seq2seq model and accuracy as my objective performance metric. I will utilize the keras deep learning library to implement my project.

The Seinfeld dialogue dataset has been downloaded from Kaggle. The data set consists diaglogue from the complete series.

https://www.kaggle.com/thec03u5/seinfeld-chronicles

The model I will use is a seq2seq, otherwise known as an encoder-decoder, model.

Risks and assumptions of data: The model that I am building uses a lot of RAM and computing power. Some strings of dialogue are too long to be useful and have to be discarded. This can negatively effect my model as the dialogue will no longer be a true back and forth conversation.

I will be using accuracy as a metric in training my encoder NN, but ultimately success will be determined subjectively, i.e. how convincing is the chatbot?

Please see my notebook, capstone-preprocess, for preliminary EDA and initial modeling. 



