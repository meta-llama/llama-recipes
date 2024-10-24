import gradio as gr
import pandas as pd
import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from lancedb.rerankers import ColbertReranker
from pathlib import Path
from PIL import Image
import io
import base64
from together import Together
import os
import logging
import argparse
import numpy as np

# Set up argument parsing
parser = argparse.ArgumentParser(description="Interactive Fashion Assistant")
parser.add_argument("--images_folder", required=True, help="Path to the folder containing compressed images")
parser.add_argument("--csv_path", required=True, help="Path to the CSV file with clothing data")
parser.add_argument("--table_path", default="~/.lancedb", help="Table path for LanceDB")
parser.add_argument("--use_existing_table", action="store_true", help="Use existing table if it exists")
parser.add_argument("--api_key", required=True, help="Together API key")
parser.add_argument("--default_model", default="BAAI/bge-large-en-v1.5", help="Default embedding model to use")
args = parser.parse_args()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

print("Starting the Fashion Assistant application...")

# Define available models
AVAILABLE_MODELS = {
    "BAAI/bge-large-en-v1.5": "bge-large-en-v1.5",
    "BAAI/bge-small-en-v1.5": "bge-small-en-v1.5",
    "BAAI/bge-reranker-base": "bge-reranker-base",
    "BAAI/bge-reranker-large": "bge-reranker-large"
}

# Define retrieval methods
RETRIEVAL_METHODS = ["Semantic Search", "Full Text Search", "Hybrid Search"]

# Connect to LanceDB
print("Connecting to LanceDB...")
db = lancedb.connect(args.table_path)

def create_table_for_model(model_name):
    print(f"Initializing the sentence transformer model: {model_name}")
    model = get_registry().get("sentence-transformers").create(name=model_name, device="cpu")
    
    class Schema(LanceModel):
        Filename: str
        Title: str
        Size: str
        Gender: str
        Description: str = model.SourceField()
        Category: str
        Type: str
        vector: Vector(model.ndims()) = model.VectorField()
    
    table_name = f"clothes_{AVAILABLE_MODELS[model_name]}"
    if not args.use_existing_table:
        tbl = db.create_table(name=table_name, schema=Schema, mode="overwrite")
        df = pd.read_csv(args.csv_path)
        df = df.dropna().astype(str)
        tbl.add(df.to_dict('records'))
        tbl.create_fts_index(["Description"], replace=True)
        print(f"Created and populated table {table_name}")
    else:
        tbl = db.open_table(table_name)
        tbl.create_fts_index(["Description"], replace=True)
        print(f"Opened existing table {table_name}")
    return tbl

tables = {model: create_table_for_model(model) for model in AVAILABLE_MODELS}
current_table = tables[args.default_model]
current_retrieval_method = "Semantic Search"

# Set up the Together API client
os.environ["TOGETHER_API_KEY"] = args.api_key
client = Together(api_key=args.api_key)
print("Together API client set up successfully.")

def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def generate_description(image):
    print("Generating description for uploaded image...")
    base64_image = encode_image(image)
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-Vision-Free",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Describe this clothing item in detail."
                        }
                    ]
                }
            ],
            max_tokens=512,
            temperature=0.7,
        )
        description = response.choices[0].message.content
        print(f"Generated description: {description}")
        return description
    except Exception as e:
        print(f"Error generating description: {e}")
        return "Error generating description"

def process_chat_input(chat_history, user_input):
    print(f"Processing chat input: {user_input}")
    messages = [
        {"role": "system", "content": "You are a helpful fashion assistant."}
    ]
    for user_msg, assistant_msg in chat_history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    
    user_input += ". START YOUR MESSAGE DIRECTLY WITH A RESPONSE LIST. DO NOT REPEAT THE NAME OF THE ITEM MENTIONED IN THE QUERY. Start your message with '1. ..' "
    messages.append({"role": "user", "content": user_input})
    print(f"Chat history: {messages}")
    try:
        bot_response = client.chat.completions.create(
            model="meta-llama/Llama-Vision-Free",
            messages=messages,
            max_tokens=512,
            temperature=0.7,
        ).choices[0].message.content
        
        print(f"Bot response: {bot_response}")
        return user_input, bot_response
    except Exception as e:
        print(f"Error processing chat input: {e}")
        return user_input, "Error processing chat input"

def retrieve_similar_items(description, n=10):
    print(f"Retrieving similar items for: {description}")
    try:
        if current_retrieval_method == "Semantic Search":
            results = current_table.search(description).limit(n).to_pandas()
        elif current_retrieval_method == "Full Text Search":
            results = current_table.search(description, query_type="fts").limit(n).to_pandas()
        elif current_retrieval_method == "Hybrid Search":
            reranker = ColbertReranker(
                model_name="answerdotai/answerai-colbert-small-v1",
                column="Description")
            results = current_table.search(description, query_type="hybrid").rerank(reranker=reranker).limit(n).to_pandas()
        else:
            raise ValueError("Invalid retrieval method")
        print(f"Retrieved {len(results)} similar items using {current_retrieval_method}.")
        return results
    except Exception as e:
        print(f"Error retrieving similar items: {e}")
        return pd.DataFrame()

def rewrite_query(original_query, item_description):
    print(f"Rewriting query: {original_query}")
    messages = [
        {"role": "system", "content": "You are a helpful fashion assistant. Rewrite the user's query to include details from the item description."},
        {"role": "user", "content": f"Item description: {item_description}"},
        {"role": "user", "content": f"User query: {original_query}"},
        {"role": "user", "content": "Please rewrite the query to include relevant details from the item description."}
    ]
    
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-Vision-Free",
            messages=messages,
            max_tokens=512,
            temperature=0.7,
        )
        rewritten_query = response.choices[0].message.content
        print(f"Rewritten query: {rewritten_query}")
        return rewritten_query
    except Exception as e:
        print(f"Error rewriting query: {e}")
        return original_query

def fashion_assistant(image, chat_input, chat_history):
    if chat_input != "":
        print("Processing chat input...")
        last_description = chat_history[-1][1] if chat_history else ""
        user_message, bot_response = process_chat_input(chat_history, chat_input)
        similar_items = retrieve_similar_items(bot_response)
        gallery_data = create_gallery_data(similar_items)
        return chat_history + [[user_message, bot_response]], bot_response, gallery_data, last_description
    elif image is not None:
        print("Processing uploaded image...")
        description = generate_description(image)
        user_message = f"I've uploaded an image. The description is: {description}"
        user_message, bot_response = process_chat_input(chat_history, user_message)
        similar_items = retrieve_similar_items(description)
        gallery_data = create_gallery_data(similar_items)
        return chat_history + [[user_message, bot_response]], bot_response, gallery_data, description
    else:
        print("No input provided.")
        return chat_history, "", [], ""

def create_gallery_data(results):
    return [
        (str(Path(args.images_folder) / row['Filename']), f"{row['Title']}\n{row['Description']}")
        for _, row in results.iterrows()
    ]

def on_select(evt: gr.SelectData):
    return f"Selected {evt.value} at index {evt.index}"

def update_chat(image, chat_input, chat_history, last_description):
    new_chat_history, last_response, gallery_data, new_description = fashion_assistant(image, chat_input, chat_history)
    if new_description:
        last_description = new_description
    return new_chat_history, new_chat_history, "", last_response, gallery_data, last_description

def update_model(model_name):
    global current_table
    current_table = tables[model_name]
    return f"Switched to model: {model_name}"

def update_retrieval_method(method):
    global current_retrieval_method
    current_retrieval_method = method
    return f"Switched to retrieval method: {method}"

# Define the Gradio interface
print("Setting up Gradio interface...")
with gr.Blocks() as demo:
    gr.Markdown("# Interactive Fashion Assistant")
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Clothing Image")
            model_dropdown = gr.Dropdown(
                choices=list(AVAILABLE_MODELS.keys()),
                value=args.default_model,
                label="Embedding Model"
            )
            retrieval_dropdown = gr.Dropdown(
                choices=RETRIEVAL_METHODS,
                value="Semantic Search",
                label="Retrieval Method"
            )
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(label="Chat History")
            chat_input = gr.Textbox(label="Chat Input")
            chat_button = gr.Button("Send")
        with gr.Column(scale=2):
            gallery = gr.Gallery(
                label="Retrieved Clothes",
                show_label=True,
                elem_id="gallery",
                columns=[5],
                rows=[2],
                object_fit="contain",
                height="auto"
            )
            selected_image = gr.Textbox(label="Selected Image")
    
    chat_state = gr.State([])
    last_description = gr.State("")
    
    image_input.change(update_chat, inputs=[image_input, chat_input, chat_state, last_description], 
                       outputs=[chat_state, chatbot, chat_input, chat_input, gallery, last_description])
    chat_button.click(update_chat, inputs=[image_input, chat_input, chat_state, last_description], 
                      outputs=[chat_state, chatbot, chat_input, chat_input, gallery, last_description])
    gallery.select(on_select, None, selected_image)
    model_dropdown.change(update_model, inputs=[model_dropdown], outputs=[gr.Textbox(label="Model Status")])
    retrieval_dropdown.change(update_retrieval_method, inputs=[retrieval_dropdown], outputs=[gr.Textbox(label="Retrieval Method Status")])

    # Disable embedding model dropdown when Hybrid Search is selected
    retrieval_dropdown.change(lambda x: gr.update(interactive=x != "Hybrid Search"), inputs=[retrieval_dropdown], outputs=[model_dropdown])

print("Gradio interface set up successfully. Launching the app...")
demo.launch()
print("Fashion Assistant application is now running!")
