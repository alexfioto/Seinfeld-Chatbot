# imports
from flask import Flask, Response, request, render_template, jsonify
from chatterbot import ChatBot

import pickle 

app = Flask('Seinfeld Chatbot')

bot = ChatBot('Flask_app',
               database_uri='sqlite:///db.sqlite3')




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