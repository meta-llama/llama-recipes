# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datasets import Dataset
import random
from langchain_community.document_loaders import SitemapLoader,DirectoryLoader
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
import copy


# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def strip_str(s: str) -> str:
    """
    Helper function for helping format strings returned by GPT-4.
    """
    l, r = 0, len(s)-1
    beg_found = False
    for i in range(len(s)):
        if s[i].isalpha():
            if not beg_found:
                l = i
                beg_found = True
            else:
                r = i
    r += 2
    return s[l:min(r, len(s))]
def clean_documents(raw_text):
    all_lines = []
    for line in raw_text.split("\n"):
        line = line.strip()
        if len(line.split()) == 0:
            continue
        else:
            all_lines.append(line)
    result = " ".join(all_lines)
    return result
def clean_text(content: BeautifulSoup) -> str:
    # Find all 'nav' and 'header' elements in the BeautifulSoup object
    nav_elements = content.find_all("nav")
    header_elements = content.find_all("header")
    mydivs = content.find_all("div", {"role": "list"})
    # Remove each 'nav' and 'header' element from the BeautifulSoup object
    for element in nav_elements + header_elements+mydivs:
        element.decompose()
    raw_text = content.get_text("\n")
    return clean_documents(raw_text)
# Read
def read_file_content(xml_path: str, data_folder: str) -> str:
    if xml_path and data_folder:
        logging.info(f"Error: both xml_path and data_folder are provided, will only read from xml for now")
    if not xml_path and not data_folder:
        logging.info(f"Error: both xml_path and data_folder are not provided")
        return ""
    if xml_path:
        if not os.path.exists(xml_path):
            logging.info(f"Error: {xml_path} does not exist")
            return ""
        # Use langchain to load the documents from webpage links in the xml file
        sitemap_loader = SitemapLoader(web_path=xml_path,is_local=True,parsing_function=clean_text)
        sitemap_loader.requests_kwargs = {"verify": False}
        docs = sitemap_loader.load()
        return docs
    elif len(data_folder) != 0:
        if not os.path.exists(data_folder):
            logging.info(f"Error: {data_folder} does not exist")
            return ""
        # Use langchain to load the documents from data folder
        loader = DirectoryLoader(data_folder)
        docs = loader.load()
        return docs



def get_chunks(
    docs: list,
    chunk_size: int = 1000,
    api_config: dict = None,
) -> list[str]:
    """
    Takes in a list of documents, breaks them down into chunks of size
    `chunk_size`, and returns the chunks.
    """
    chunks = []
    if  len(docs) == 0:
        raise TypeError("Can not get chunks from empty text")
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=api_config["chunk_size"],chunk_overlap=int(api_config["chunk_size"] / 10),separators= ["----------","\n\n", "\n", " "],strip_whitespace=True)
        docs_processed = text_splitter.split_documents(docs)
        logging.info(f"Total number of docs_processed: {len(docs_processed)}")
        # Remove duplicates
        unique_texts = {}
        docs_processed_unique = []
        for doc in docs_processed:
            if doc.page_content not in unique_texts and len(doc.page_content) > 100 :
                unique_texts[doc.page_content] = True
                docs_processed_unique.append(doc)        
        chunks = [chunk.page_content for chunk in docs_processed_unique]
        logging.info(f"Total number of docs_processed_unique: {len(docs_processed_unique)}")
    return chunks
# read all the files in the data folder, then split them into chunks
# generate questions for each chunk and return zip of chunk and related questions list
def generate_questions(api_config):
    # get documents from the data folder or xml file
    api_url = api_config["endpoint_url"]
    key = api_config["api_key"]
    documents = read_file_content(api_config["xml_path"],api_config["data_dir"])
    if len(documents) == 0:
        logging.info(f"Error reading files, document_text is {len(documents)}")
    document_batches = get_chunks(documents,api_config["chunk_size"],api_config)
    # use OpenAI API protocol to hanlde the chat request, including local VLLM openai compatible server
    llm = ChatOpenAI(
        openai_api_key=key,
        openai_api_base=api_url,
        model_name=api_config["model"],
        temperature=0.0,
        max_tokens=500
        )
    all_tasks = [api_config['question_prompt_template'].format(num_questions=str(api_config['questions_per_chunk']),context=document) for document in document_batches]
    generated_answers = llm.batch(all_tasks)
    generated_answers = [ item.content for item in generated_answers]
    if len(generated_answers) == 0:
        logging.error("No model answers generated. Please check the input context or model configuration in ",api_config["model"])
        return []
    final_result = []
    for result in generated_answers:
        queries = result.split('\n')
        queries = [strip_str(q) for q in queries]
        queries = [q for q in queries if any(c.isalpha() for c in q)]
        if len(queries) > int(api_config['questions_per_chunk']):
            # As the model may have unrelated question at the begining of the result
            # if queries is more than questions_per_chunk, then we need to truncate it and only keep last questions_per_chunk lines
            queries = queries[-int(api_config['questions_per_chunk']):]
        final_result.append(queries)
    return list(zip(document_batches,final_result))

# Generate COT answer for each question given the chunk context
def generate_COT(chunk_questions_zip,api_config) -> dict:
    all_tasks = []
    chunk_questions = []
    question_asked = set()
    for document_content,questions in chunk_questions_zip:
        for question in questions:
            question = question.strip()
            # avoid asking the same question twice
            if question not in question_asked:
                question_asked.add(question)
                prompt = api_config['COT_prompt_template'].format(question=question,context=str(document_content))
                all_tasks.append(prompt)
                chunk_questions.append((document_content,question))
    # use OpenAI API protocol to hanlde the chat request, including local VLLM openai compatible server
    llm = ChatOpenAI(
        openai_api_key=api_config["api_key"],
        openai_api_base=api_config["endpoint_url"],
        model_name=api_config["model"],
        temperature=0.0,
        max_tokens=500
        )
    generated_answers = llm.batch(all_tasks)
    generated_answers = [ item.content for item in generated_answers]
    COT_results = []
    # return a list of (chunk, question, generated_answer)
    for (chunk, question),generated_answer in zip(chunk_questions,generated_answers):
        COT_results.append((chunk,question,generated_answer))
    return COT_results

def add_chunk_to_dataset(
    chunk_questions_zip: list,
    api_config: dict,
) -> None:
    """
    Given a chunk and related questions lists, create {Q, A, D} triplets and add them to the dataset.
    """
    num_distract = api_config["num_distract_docs"]
    p = api_config["refusal_probability"]
    chunks = [chunk for chunk, _ in chunk_questions_zip]
    COT_results = generate_COT(chunk_questions_zip,api_config)
    logging.info(f"COT generation completed, total num of COT results: {len(COT_results)}")
    completed,refusal= 0,0
    data_list = []
    for chunk, q , cot in COT_results:
        # The COT answer will be used as the label in the fine-tuning stage

        datapt = {
            "id": None,
            "type": "general",
            "question": q,
            "context": None,
            "oracle_context": None,
            "cot_answer": cot
        }
        i = chunks.index(chunk)
        datapt["id"] = f"seed_task_{len(data_list)}"
        # add num_distract distractor docs
        docs = [chunk]
        indices = list(range(0, len(chunks)))
        indices.remove(i)
        for j in random.sample(indices, num_distract):
            docs.append(chunks[j])
        doc_copy = docs.copy()
        random.shuffle(docs)
        d = {
            "title": [],
            "sentences": []
        }

        d["title"].append(["placeholder_title"]*(num_distract+1))
        d["sentences"].append(docs)
        datapt["context"] = d
        datapt["oracle_context"] = chunk

        # construct model instruction
        context = ""
        for doc in docs:
            context += "<DOCUMENT>" + str(doc) + "</DOCUMENT>\n"
        context += q
        # This instruction will be used in the fine-tuning stage
        datapt["instruction"] = context
        datapt_copy = copy.deepcopy(datapt)
        # add to dataset
        data_list.append(datapt)
        # decides whether to add refusal example where the related documents are not provided
        refusal = random.uniform(0, 1) <= p
        if refusal:
            doc_copy[0] = chunks[random.sample(indices, 1)[0]]
            random.shuffle(doc_copy)
            refusl_context = ""
            for doc in doc_copy:
                refusl_context += "<DOCUMENT>" + str(doc) + "</DOCUMENT>\n"
            refusl_context += q
            # This instruction will be used in the fine-tuning stage
            datapt_copy["id"] = f"refusal_task_{len(data_list)}"
            datapt_copy["instruction"] = refusl_context
            datapt_copy["cot_answer"] = "Sorry, I don't know the answer to this question because related documents are not found. Please try again."
            data_list.append(datapt_copy)
            refusal += 1
        completed += 1
        if completed % 100 == 0:
            logging.info(f"refusal example added: {refusal}, total examples added: {completed}, total examples to be added: {len(COT_results)- completed}")
    ds = Dataset.from_list(data_list)
    return ds
