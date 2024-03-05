# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import streamlit as st
from langchain.llms import Replicate
import os

st.title("Llama2-powered Streamlit App")

with st.sidebar:
    os.environ["REPLICATE_API_TOKEN"] = "<your replicate api token>"

def generate_response(input_text):
    llama2_13b_chat = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"

    llm = Replicate(
        model=llama2_13b_chat,
        model_kwargs={"temperature": 0.01, "top_p": 1, "max_new_tokens":500}
    )
    st.info(llm(input_text))

with st.form("my_form"):
    text = st.text_area("Enter text:", "What is Generative AI?")
    submitted = st.form_submit_button("Submit")
    generate_response(text)
