## Llama 3.2 Vision Models Fine-Tuning Recipe
This recipe steps you through how to finetune a Llama 3.2 vision model on the OCR VQA task using the [OCRVQA](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron/viewer/ocrvqa?row=0) dataset.

**Disclaimer**: As our vision models already have a very good OCR ability, here we use the OCRVQA dataset only for demonstration purposes of the required steps for fine-tuning our vision models with llama-recipes.

### Fine-tuning steps

We created an example script [ocrvqa_dataset.py](./datasets/ocrvqa_dataset.py) that can load the OCRVQA dataset with `get_custom_dataset` function, then provide OCRVQADataCollator class to process the image dataset.

For **full finetuning with FSDP**, we can run the following code:

```bash
  torchrun --nnodes 1 --nproc_per_node 4  recipes/quickstart/finetuning/finetuning.py --enable_fsdp --lr 1e-5  --num_epochs 3 --batch_size_training 2 --model_name meta-llama/Llama-3.2-11B-Vision-Instruct --dist_checkpoint_root_folder ./finetuned_model --dist_checkpoint_folder fine-tuned  --use_fast_kernels --dataset "custom_dataset" --custom_dataset.test_split "test" --custom_dataset.file "recipes/quickstart/finetuning/datasets/ocrvqa_dataset.py"  --run_validation True --batching_strategy padding
```

For **LoRA finetuning with FSDP**, we can run the following code:

```bash
  torchrun --nnodes 1 --nproc_per_node 4  recipes/quickstart/finetuning/finetuning.py --enable_fsdp --lr 1e-5  --num_epochs 3 --batch_size_training 2 --model_name meta-llama/Llama-3.2-11B-Vision-Instruct --dist_checkpoint_root_folder ./finetuned_model --dist_checkpoint_folder fine-tuned  --use_fast_kernels --dataset "custom_dataset" --custom_dataset.test_split "test" --custom_dataset.file "recipes/quickstart/finetuning/datasets/ocrvqa_dataset.py"  --run_validation True --batching_strategy padding  --use_peft --peft_method lora
```
**Note**: `--batching_strategy padding` is needed as the vision model will not work with `packing` method.

For more details about the finetuning configurations, please read the [finetuning readme](./README.md).

For more details about local inference with the fine-tuned checkpoint, please read [Inference with FSDP checkpoints section](https://github.com/meta-llama/llama-recipes/tree/main/recipes/quickstart/inference/local_inference#inference-with-fsdp-checkpoints) to learn how to convert the FSDP weights into a consolidated Hugging Face formatted model for local inference.

### How to use a custom dataset to fine-tune vision model

In order to use a custom dataset, please follow the steps below:

1. Create a new dataset python file under `recipes/quickstart/finetuning/dataset` folder.
2. In this python file, you need to define a `get_custom_dataset(dataset_config, processor, split, split_ratio=0.9)` function that handles the data loading.
3. In this python file, you need to define a `get_data_collator(processor)` function that returns a custom data collator that can be used by the Pytorch Data Loader.
4. This custom data collator class must have a `__call__(self, samples)` function that converts the image and text samples into the actual inputs that vision model expects.
5. Run the `torchrun` command from above section, please change the `--custom_dataset.file` to the new dataset python file, adjust the learning rate accordingly.
