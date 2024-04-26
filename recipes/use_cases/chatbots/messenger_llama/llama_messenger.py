# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 3 Community License Agreement.

import langchain
from langchain.llms import Replicate

from flask import Flask
from flask import request
import os
import requests
import json

os.environ["REPLICATE_API_TOKEN"] = "<your replicate api token>"
llama3_8b_chat = "meta/meta-llama-3-8b-instruct"

llm = Replicate(
    model=llama3_8b_chat,
    model_kwargs={"temperature": 0.0, "top_p": 1, "max_new_tokens":500}
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
        'access_token': "<your page access token>"
    }
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(url, params=params, headers=headers)
    print(response.status_code)
    print(response.text)

    return message + "<p/>" + answer

