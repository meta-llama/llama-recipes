# FAQ

Here we discuss frequently asked questions that may occur and we found useful along the way.

1. Does FSDP support mixed precision in one FSDP unit? Meaning, in one FSDP unit some of the parameters are in Fp16/Bf16 and others in FP32.

    FSDP requires each FSDP unit to have consistent precision, so this case is not supported at this point. It might be added in future but no ETA at the moment.

2.  How does FSDP handles mixed grad requirements?

    FSDP does not support mixed `require_grad` in one FSDP unit. This means if you are planning to freeze some layers, you need to do it on the FSDP unit level rather than model layer. For example, let us assume our model has 30 decoder layers and we want to freeze the bottom 28 layers and only train 2 top transformer layers. In this case, we need to make sure `require_grad` for the top two transformer layers are set to `True`.

3. How do PEFT methods work with FSDP in terms of grad requirements/layer freezing?

    We wrap the PEFT modules separate from the transformer layer in auto_wrapping policy, that would result in PEFT models having `require_grad=True` while the rest of the model is  `require_grad=False`.

4. Can I add custom datasets?

    Yes, you can find more information on how to do that [here](Dataset.md).

5. What are the hardware SKU requirements for deploying these models?

    Hardware requirements vary based on latency, throughput and cost constraints. For good latency, the models were split across multiple GPUs with tensor parallelism in a machine with NVIDIA A100s or H100s. But TPUs, other types of GPUs like A10G, T4, L4, or even commodity hardware can also be used to deploy these models (e.g. https://github.com/ggerganov/llama.cpp).
    If working on a CPU, it is worth looking at this [blog post](https://www.intel.com/content/www/us/en/developer/articles/news/llama2.html) from Intel for an idea of Llama 2's performance on a CPU.

6. What are the hardware SKU requirements for fine-tuning Llama pre-trained models?

    Fine-tuning requirements vary based on amount of data, time to complete fine-tuning and cost constraints. To fine-tune these models we have generally used multiple NVIDIA A100 machines with data parallelism across nodes and a mix of data and tensor parallelism intra node. But using a single machine, or other GPU types like NVIDIA A10G or H100 are definitely possible (e.g. alpaca models are trained on a single RTX4090: https://github.com/tloen/alpaca-lora).

7. How to handle CUDA memory fragmentations during fine-tuning that may lead into an OOM? In some cases you may experience that after model checkpointing specially with FSDP (this usually does not happen with PEFT methods), the reserved and allocated CUDA memory has increased. This might be due to CUDA memory fragmentations. PyTorch recenly added an enviroment variable that helps to better manage memory fragmentation (this feature in available on PyTorch nightlies at the time of writing this doc July 30 2023). You can set this in your main training script as follows: 

```bash

os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'

```
We also added this enviroment variable in `setup_environ_flags` of the [train_utils.py](../src/llama_recipes/utils/train_utils.py), feel free to uncomment it if required.

8. Additional debugging flags? the environment variable `TORCH_DISTRIBUTED_DEBUG` can be used to trigger additional useful logging and collective synchronization checks to ensure all ranks are synchronized appropriately. `TORCH_DISTRIBUTED_DEBUG` can be set to either OFF (default), INFO, or DETAIL depending on the debugging level required. Please note that the most verbose option, DETAIL may impact the application performance and thus should only be used when debugging issues.

We also added this enviroment variable in `setup_environ_flags` of the [train_utils.py](../src/llama_recipes/utils/train_utils.py), feel free to uncomment it if required.