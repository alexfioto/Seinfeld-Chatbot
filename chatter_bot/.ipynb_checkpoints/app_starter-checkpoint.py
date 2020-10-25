# imports
from flask import Flask, Response, request, render_template, jsonify
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer, ListTrainer
from numpy.random import choice

import pickle 

app = Flask('Seinfeld Chatbot')

bot = ChatBot('Seinfeld')
corpus_trainer = ChatterBotCorpusTrainer(bot)
#corpus_trainer.train('chatterbot.corpus.english')


with open('../combos.pkl', 'rb') as f:
    combos = pickle.load(f)
    
def random_seinfeld(n):
    result = []
    for i in range(n):
        result.append(combos[choice(len(combos))])
    return result

random_combos = random_seinfeld(2500)

trainer = ListTrainer(bot)
for combo in random_combos:
    trainer.train(combo)



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/get')
def get_bot_response():
    user_text = request.args.get('msg')
    return str(bot.get_response(user_text))



# Call app.run(debug=True) when python script is called
if __name__ == '__main__':
    app.run(debug=True)