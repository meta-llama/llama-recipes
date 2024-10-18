# End to End Showcase using Llama models for Multi-Modal RAG 

## Story Overview: Multi-Modal RAG using `Llama-3.2-11B` model: 

- **Data Labelling and Preperation:** We start by downloading 5000 images of clothing items and labelling them using 11B model
- **Clearning Labels:** With the labels based on the notebook above, we will then clean the dataset and prepare it for RAG
- **Building Vector DB and RAG Pipeline:** With the final clean dataset, we can use descriptions and 11B model to generate recommendations

## Resources used: 

List of models and libraries used in the showcase:

- [Llama-3.2-11B-Vision-Instruct](https://www.llama.com/docs/how-to-guides/vision-capabilities/) Model
- [Lance-db for vector database](https://lancedb.com)
- [This]() Kaggle dataset for building our work
- [HF Dataset](https://huggingface.co/datasets/Sanyam/MM-Demo) Since output of the model can be non-deterministic everytime we run, we will use the uploaded dataset to give a universal experience

## Detailed Outline 

Here's the detailed outline:

Step 1: Data Prep and Synthetic Labeling:

The dataset consists of 5000 images with some classification.

The first half is preparing the dataset for labelling:
- Clean/Remove corrupt images
- EDA to understand existing distribution
- Merging up categories of clothes to reduce complexity 
- Balancing dataset by randomly sampling images

Second Half consists of Labelling the dataset. We are bound by an interesting constraint here, 11B model can only caption one image at a time:
- We load a few images and test captioning
- We run this pipeline on random images and iterate on the prompt till we feel the model is giving good outputs
- Finally, we can create a script to label all 5000 images on multi-GPU

After running the script on the entire dataset, we have more data cleaning to perform:

- Step 2: Cleaning up Synthetic Labels and preparing the dataset
- Step 3: Notebook 3: MM-RAG using lance-db to validate idea
- Step 4: Gradio App using Together API for Llama-3.2-11B and Lance-db for RAG