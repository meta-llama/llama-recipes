# End to End Tutorial on using Llama models for Multi-Modal RAG 

## Recipe Overview: Multi-Modal RAG using `Llama-3.2-11B` model: 

This is a complete workshop on how to label images using the new Llama 3.2-Vision Models and performing RAG using the image caption capabilities of the model.

- **Data Labeling and Preparation:** We start by downloading 5000 images of clothing items and labeling them using `Llama-3.2-11B-Vision-Instruct` model
- **Cleaning Labels:** With the labels based on the notebook above, we will then clean the dataset and prepare it for RAG
- **Building Vector DB and RAG Pipeline:** With the final clean dataset, we can use descriptions and 11B model to generate recommendations

## Requirements:

Before we start:

1. Please grab your HF CLI Token from [here](https://huggingface.co/settings/tokens)
2. Git clone [this dataset](https://huggingface.co/datasets/Sanyam/MM-Demo) inside the Multi-Modal-RAG folder: `git clone https://huggingface.co/datasets/Sanyam/MM-Demo` (Remember to thank the original author by up voting [Kaggle Dataset](https://www.kaggle.com/datasets/agrigorev/clothing-dataset-full))
3. Make sure you grab a together.ai token [here](https://www.together.ai)

## Detailed Outline for running:

Order of running files, the notebook establish the method of approaching the problem. Once we establish it, we use the scripts to run the method end to end.

- Notebook 1: `Part_1_Data_Preparation.ipynb`
- Script: `label_script.py`
- Notebook 2: `Part_2_Cleaning_Data_and_DB.ipynb`
- Notebook 3: `Part_3_RAG_Setup_and_Validation.ipynb`
- Script: `final_demo.py`

Here's the detailed outline:

### Step 1: Data Prep and Synthetic Labeling:

In this step we start with an unlabeled dataset and use the image captioning capability of the model to write a description of the image and categorize it.

[Notebook for Step 1](./notebooks/Part_1_Data_Preparation.ipynb) and [Script for Step 1](./scripts/label_script.py)

To run the script (remember to set n):
```
python scripts/label_script.py --hf_token "your_huggingface_token_here" \
    --input_path "../MM-Demo/images_compressed" \
    --output_path "../MM-Demo/output/" \
    --num_gpus N
```

The dataset consists of 5000 images with some meta-data.

The first half is preparing the dataset for labeling:
- Clean/Remove corrupt images
- Some exploratory analysis to understand existing distribution
- Merging up categories of clothes to reduce complexity 
- Balancing dataset by randomly sampling images to have an equal distribution for retrieval

Second Half consists of Labeling the dataset. Llama 3.2, 11B model can only process one image at a time:
- We load a few images and test captioning
- We run this pipeline on random images and iterate on the prompt till we feel the model is giving good outputs
- Finally, we can create a script to label all 5000 images on multi-GPU

After running the script on the entire dataset, we have more data cleaning to perform.

### Step 2: Cleaning up Synthetic Labels and preparing the dataset:

[Notebook for Step 2](./notebooks/Part_2_Cleaning_Data_and_DB.ipynb)

We notice that even after some fun prompt engineering, the model faces some hallucinations-there are some issues with the JSON formatting and we notice that it hallucinates the label categories. Here is how we address this:

- Re-balance the dataset by mapping correct categories. This is useful to make sure we have an equal distribution in our dataset for retrieval
- Fix Descriptions so that we can create a CSV

Now, we are ready to try our vector db pipeline:

### Step 3: Notebook 3: MM-RAG using lance-db to validate idea

[Notebook for Step 3](./notebooks/Part_3_RAG_Setup_and_Validation.ipynb) and [Final Demo Script](./scripts/label_script.py)


With the cleaned descriptions and dataset, we can now store these in a vector-db, here's the steps:


- We create embeddings using the text description of our clothes
- Use 11-B model to describe the uploaded image
- Ask the model to suggest complementary items to the upload
- Try to find similar or complementary images based on the upload

We try the approach with different retrieval methods.

Finally, we can bring this all together in a Gradio App. 

For running the script:
```
python scripts/final_demo.py \
    --images_folder "../MM-Demo/compressed_images" \
    --csv_path "../MM-Demo/final_balanced_sample_dataset.csv" \
    --table_path "~/.lancedb" \
    --api_key "your_together_api_key" \
    --default_model "BAAI/bge-large-en-v1.5" \
    --use_existing_table 
```

Note: We can further improve the description prompt. You will notice sometimes the description starts with the title of the cloth which causes in retrieval of "similar" clothes instead of "complementary" items

- Upload an image
- 11B model describes the image
- We retrieve complementary clothes to wear based on the description
- You can keep the loop going by chatting with the model

## Resources used: 

Credit and Thanks to List of models and resources used in the showcase:

Firstly, thanks to the author here for providing this dataset on which we base our exercise [here](https://www.kaggle.com/datasets/agrigorev/clothing-dataset-full)

- [Llama-3.2-11B-Vision-Instruct Model](https://www.llama.com/docs/how-to-guides/vision-capabilities/)
- [Lance-db for vector database](https://lancedb.com)
- [This Kaggle dataset](https://www.kaggle.com/datasets/agrigorev/clothing-dataset-full)
- [HF Dataset](https://huggingface.co/datasets/Sanyam/MM-Demo) Since output of the model can be non-deterministic every time we run, we will use the uploaded dataset to give a universal experience
- [Together API for demo](https://www.together.ai)