# End to End Tutorial on using Llama models for Multi-Modal RAG 

## Recipe Overview: Multi-Modal RAG using `Llama-3.2-11B` model: 

This is a complete workshop on labelling images using the new Llama 3.2-Vision Models and performing RAG using the image caption capiblites of the model.

- **Data Labeling and Preparation:** We start by downloading 5000 images of clothing items and labeling them using `Llama-3.2-11B-Vision-Instruct` model
- **Cleaning Labels:** With the labels based on the notebook above, we will then clean the dataset and prepare it for RAG
- **Building Vector DB and RAG Pipeline:** With the final clean dataset, we can use descriptions and 11B model to generate recommendations

## Requirements:

Before we start:

1. Please grab your HF CLI Token from [here](https://huggingface.co/settings/tokens)
2. git clone [this dataset](https://huggingface.co/datasets/Sanyam/MM-Demo) inside the Multi-Modal-RAG folder: `git clone https://huggingface.co/datasets/Sanyam/MM-Demo`
3. Launch jupyter notebook inside this folder
4. We will also run two scripts after the notebooks
5. Make sure you grab a together.ai token [here](https://www.together.ai)

## Detailed Outline for running:

Order of running files, the notebook establish the method of approaching the problem. Once we establish it, we use the scripts to run the method end to end.

- Notebook 1: `Part_1_Data_Preperation.ipynb`
- Script: `label_script.py`
- Notebook 2: `Part_2_Cleaning_Data_and_DB.ipynb`
- Notebook 3: `Part_3_RAG_Setup_and_Validation.ipynb`
- Script: `final_demo.py`

Here's the detailed outline:

### Step 1: Data Prep and Synthetic Labeling:

[Notebook for Step 1](./notebooks/Part_1_Data_Preperation.ipynb) and [Script for Step 1](./scripts/label_script.py)

To run the script: 
```
python scripts/caption_generator.py --hf_token "your_huggingface_token_here" \
    --input_path "../images" \
    --output_path "/path/to/output/folder" \
    --num_gpus 2
```

The dataset consists of 5000 images with some meta-data.

The first half is preparing the dataset for labeling:
- Clean/Remove corrupt images
- EDA to understand existing distribution
- Merging up categories of clothes to reduce complexity 
- Balancing dataset by randomly sampling images

Second Half consists of Labeling the dataset. We are bound by an interesting constraint here, 11B model can only caption one image at a time:
- We load a few images and test captioning
- We run this pipeline on random images and iterate on the prompt till we feel the model is giving good outputs
- Finally, we can create a script to label all 5000 images on multi-GPU

After running the script on the entire dataset, we have more data cleaning to perform.

### Step 2: Cleaning up Synthetic Labels and preparing the dataset:

[Notebook for Step 2](./notebooks/Part_2_Cleaning_Data_and_DB.ipynb)

Even after our lengthy (apart from other things) prompt, the model still hallucinates categories and label-we need to address this

- Re-balance the dataset by mapping correct categories
- Fix Descriptions so that we can create a CSV

Now, we are ready to try our vector db pipeline:

### Step 3: Notebook 3: MM-RAG using lance-db to validate idea

[Notebook for Step 3](./notebooks/Part_3_RAG_Setup_and_Validation.ipynb) and [Final Demo Script](./scripts/label_script.py)

With the cleaned descriptions and dataset, we can now store these in a vector-db

You will note that we are not using the categorization from our model-this is by design to show how RAG can simplify a lot of things. 

- We create embeddings using the text description of our clothes
- Use 11-B model to describe the uploaded image
- Try to find similar or complimentary images based on the upload

We try the approach with different retrieval methods.


### Step 4: Gradio App using Together API for Llama-3.2-11B and Lance-db for RAG

Finally, we can bring this all together in a Gradio App. 

Task: We can further improve the description prompt. You will notice sometimes the description starts with the title of the cloth which causes in retrieval of "similar" clothes instead of "complementary" items

- Upload an image
- 11B model describes the image
- We retrieve complementary clothes to wear based on the description
- You can keep the loop going by chatting with the model

## Resources used: 

Credit and Thanks to List of models and resources used in the showcase:

Firstly, thanks to the author here for providing this dataset on which we base our exercise []()

- [Llama-3.2-11B-Vision-Instruct Model](https://www.llama.com/docs/how-to-guides/vision-capabilities/)
- [Lance-db for vector database](https://lancedb.com)
- [This Kaggle dataset]()
- [HF Dataset](https://huggingface.co/datasets/Sanyam/MM-Demo) Since output of the model can be non-deterministic every time we run, we will use the uploaded dataset to give a universal experience
- [Together API for demo](https://www.together.ai)