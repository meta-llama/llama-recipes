import langchain
from langchain.llms import Replicate

from flask import Flask
from flask import request
import os
import requests
import json

os.environ["REPLICATE_API_TOKEN"] = "r8_dR6bALmiSCZCZRs3JKuxkMYxkEW8b2Z0oDwCm"    
llama2_13b_chat = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"

llm = Replicate(
    model=llama2_13b_chat,
    model_kwargs={"temperature": 0.01, "top_p": 1, "max_new_tokens":500}
)

app = Flask(__name__)

@app.route('/msgrcvd_pager', methods=['POST', 'GET'])
def msgrcvd_pager():    
    message = request.args.get('message')
    sender = request.args.get('sender')
    recipient = request.args.get('recipient')

    answer = llm(message)
    print(message)
    print(answer)

    url = f"https://graph.facebook.com/v18.0/{recipient}/messages"
    params = {
        'recipient': '{"id": ' + sender + '}',
        'message': json.dumps({'text': answer}),
        'messaging_type': 'RESPONSE',
        'access_token': 'EAAEox5Brim0BOzT7xduQmLPmV5JEYC0wyfZBPE308kOPOUr02GITwIeABUT0ffvoHm2ktusKfXgwoZAQiaI6ZAobAhtGQjsYsm7VzCbVBLQjzKSMyKlmI2ZCFtZAZAEuYZCIZC2YMlCpBhjTbr1Tr7HC7Eom7EPchFpOWAGWktN1PCik17Q1KWCD1ZAdSLBQS6T1Jk4wmZA54eO3MCgQZDZ'
    }
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(url, params=params, headers=headers)
    print(response.status_code)
    print(response.text)

    return message + "<p/>" + answer

