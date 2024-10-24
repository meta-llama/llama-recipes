# End to End Showcase using Llama models for Multi-Modal RAG 

## Story Overview: Multi-Modal RAG using `Llama-3.2-11B` model: 

- **Data Labeling and Preparation:** We start by downloading 5000 images of clothing items and labeling them using 11B model
- **Cleaning Labels:** With the labels based on the notebook above, we will then clean the dataset and prepare it for RAG
- **Building Vector DB and RAG Pipeline:** With the final clean dataset, we can use descriptions and 11B model to generate recommendations

## Resources used: 

Credit and Thanks to List of models and resources used in the showcase:

Firstly, thanks to the author here for providing this dataset on which we base our exercise []()

- [Llama-3.2-11B-Vision-Instruct](https://www.llama.com/docs/how-to-guides/vision-capabilities/) Model
- [Lance-db for vector database](https://lancedb.com)
- [This]() Kaggle dataset for building our work
- [HF Dataset](https://huggingface.co/datasets/Sanyam/MM-Demo) Since output of the model can be non-deterministic every time we run, we will use the uploaded dataset to give a universal experience
- [Transformers for 11B model](https://github.com/huggingface/transformers) 
- [Gradio for Demo](https://github.com/gradio-app/gradio)
- [Together API for demo](https://www.together.ai)

## Detailed Outline 

Here's the detailed outline:

### Step 1: Data Prep and Synthetic Labeling:

The dataset consists of 5000 images with some classification.

The first half is preparing the dataset for labeling:
- Clean/Remove corrupt images
- EDA to understand existing distribution
- Merging up categories of clothes to reduce complexity 
- Balancing dataset by randomly sampling images

Second Half consists of Labeling the dataset. We are bound by an interesting constraint here, 11B model can only caption one image at a time:
- We load a few images and test captioning
- We run this pipeline on random images and iterate on the prompt till we feel the model is giving good outputs
- Finally, we can create a script to label all 5000 images on multi-GPU

After running the script on the entire dataset, we have more data cleaning to perform:

### Step 2: Cleaning up Synthetic Labels and preparing the dataset:

Even after our lengthy (apart from other things) prompt, the model still hallucinates categories and label-we need to address this

- Re-balance the dataset by mapping correct categories
- Fix Descriptions so that we can create a CSV

Now, we are ready to try our vector db pipeline:

### Step 3: Notebook 3: MM-RAG using lance-db to validate idea

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