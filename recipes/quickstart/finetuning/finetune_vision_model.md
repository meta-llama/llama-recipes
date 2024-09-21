## Fine-Tuning Meta Llama Multi Modal Models recipe
Here we discuss fine-tuning Meta Llama 3.2 11B and 90B models.

### Concepts
Model Architecture
Our Meta Llama 3.2 11B and 90B models consist of two main components: (1) an image encoder, (2) an image adapter.

[Model Architecture PICTURE]

We need have a new processor class added, that will handle the image processing and text tokenization. A processor example looks like this:



### Fine-tuning steps
1. Download the dataset:
an example of the dataset looks like this:
2. Processor example looks like this

3. Load the dataset

Full-finetune
```bash
  torchrun --nnodes 1 --nproc_per_node 8  recipes/quickstart/finetuning/finetuning.py --enable_fsdp --lr 1e-5 --context_length 8192 --num_epochs 3 --batch_size_training 2 --model_name nltpt/Llama-3.2-11B-Vision-Instruct --dist_checkpoint_root_folder /home/kaiwu/work/fb_connect/finetune_11bmodel --dist_checkpoint_folder fine-tuned  --use_fast_kernels --dataset "custom_dataset" --custom_dataset.test_split "test" --custom_dataset.file "recipes/quickstart/finetuning/datasets/vqa_dataset.py"  --run_validation True --batching_strategy padding  --use-wandb
```

LoRA:
```bash
  torchrun --nnodes 1 --nproc_per_node 4  recipes/quickstart/finetuning/finetuning.py --enable_fsdp --lr 1e-5 --context_length 8192 --num_epochs 1 --batch_size_training 1 --model_name llava-hf/llama3-llava-next-8b-hf --dist_checkpoint_root_folder /home/kaiwu/work/fb_connect/finetune_model --dist_checkpoint_folder fine-tuned  --use_fast_kernels --dataset "custom_dataset" --custom_dataset.test_split "test" --custom_dataset.file "recipes/quickstart/finetuning/datasets/vqa_dataset.py" --use-wandb  --run_validation True
```
