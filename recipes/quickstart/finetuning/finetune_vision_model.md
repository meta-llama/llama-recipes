## Fine-Tuning Meta Llama Multi Modal Models recipe
This recipe steps you through how to finetune a Llama 3.2 vision model on the VQA task using the [the_cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) dataset.

### Concepts
Model Architecture
Our Meta Llama 3.2 11B and 90B models consist of two main components: (1) an image encoder, (2) an image adapter.

[Model Architecture PICTURE]

We need have a new processor class added, that will handle the image processing and text tokenization. A processor example looks like this:



### Fine-tuning steps


For **full finetuning with FSDP**, we can run the following code:
```bash
  torchrun --nnodes 1 --nproc_per_node 4  recipes/quickstart/finetuning/finetuning.py --enable_fsdp --lr 1e-5 --context_length 8192 --num_epochs 3 --batch_size_training 2 --model_name meta-llama/Llama-3.2-11B-Vision-Instruct --dist_checkpoint_root_folder ./finetuned_model --dist_checkpoint_folder fine-tuned  --use_fast_kernels --dataset "custom_dataset" --custom_dataset.test_split "test" --custom_dataset.file "recipes/quickstart/finetuning/datasets/vqa_dataset.py"  --run_validation True --batching_strategy padding
```

For **LoRA finetuning with FSDP**, we can run the following code:
```bash
  torchrun --nnodes 1 --nproc_per_node 4  recipes/quickstart/finetuning/finetuning.py --enable_fsdp --lr 1e-5 --context_length 8192 --num_epochs 3 --batch_size_training 2 --model_name meta-llama/Llama-3.2-11B-Vision-Instruct --dist_checkpoint_root_folder ./finetuned_model --dist_checkpoint_folder fine-tuned  --use_fast_kernels --dataset "custom_dataset" --custom_dataset.test_split "test" --custom_dataset.file "recipes/quickstart/finetuning/datasets/vqa_dataset.py"  --run_validation True --batching_strategy padding  --use_peft --peft_method lora
```
**Note**: `--batching_strategy padding` is needed as the vision model will not work with `packing` method.

For more details about the finetuning configurations, please read the [finetuning readme](./README.md).

### How to use custom dataset to fine-tune vision model

1. Create a new dataset python file under `recipes/quickstart/finetuning/dataset` folder
2. In this python file, you need to define a `get_custom_dataset(dataset_config, processor, split, split_ratio=0.9)` function that handles the dataloading.
3. In this python file, you need to define a `get_data_collator(processor)` that returns a custom data collartor that can be used by the Pytorch Data Loader.
4. This custom data collator class must have a `__call__(self, samples)` function that converts the image and text samples into the actual inputs that vision model expects.
