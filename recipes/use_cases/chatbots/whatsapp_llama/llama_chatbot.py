# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 3 Community License Agreement.

import langchain
from langchain.llms import Replicate

from flask import Flask
from flask import request
import os
import requests
import json

class WhatsAppClient:

    API_URL = "https://graph.facebook.com/v17.0/"
    WHATSAPP_API_TOKEN = "<Temporary access token from your WhatsApp API Setup>"
    WHATSAPP_CLOUD_NUMBER_ID = "<Phone number ID from your WhatsApp API Setup>"

    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {self.WHATSAPP_API_TOKEN}",
            "Content-Type": "application/json",
        }
        self.API_URL = self.API_URL + self.WHATSAPP_CLOUD_NUMBER_ID

    def send_text_message(self,message, phone_number):
        payload = {
            "messaging_product": 'whatsapp',
            "to": phone_number,
            "type": "text",
            "text": {
                "preview_url": False,
                "body": message
            }
        }
        response = requests.post(f"{self.API_URL}/messages", json=payload,headers=self.headers)
        print(response.status_code)
        assert response.status_code == 200, "Error sending message"
        return response.status_code

os.environ["REPLICATE_API_TOKEN"] = "<your replicate api token>"    
llama3_8b_chat = "meta/meta-llama-3-8b-instruct"

llm = Replicate(
    model=llama3_8b_chat,
    model_kwargs={"temperature": 0.0, "top_p": 1, "max_new_tokens":500}
)
client = WhatsAppClient()
app = Flask(__name__)

@app.route("/")
def hello_llama():
    return "<p>Hello Llama 3</p>"

@app.route('/msgrcvd', methods=['POST', 'GET'])
def msgrcvd():    
    message = request.args.get('message')
    answer = llm(message)
    print(message)
    print(answer)
    client.send_text_message(llm(message), "<your phone number>")
    return message + "<p/>" + answer

