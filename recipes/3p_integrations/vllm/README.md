# Llama inference with vLLM

This folder contains an example for running Llama inference on multiple-gpus in single- as well as multi-node scenarios using vLLM.

## Prerequirements

To run this example we will need to install vLLM as well as ray in case multi-node inference is the goal.

```bash
pip install vllm

# For multi-node inference we also need to install ray
pip install ray[default]
```

For the following examples we will assume that we fine-tuned a base model using the LoRA method and we have setup the following environment variables pointing to the base model as well as LoRA adapter:

```bash
export MODEL_PATH=/path/to/out/base/model
export PEFT_MODEL_PATH=/path/to/out/peft/model
```

## Single-node multi-gpu inference
To launch the inference simply execute the following command changing the tp_size parameter to the numbers of GPUs you have available:

``` bash
python inference.py --model_name $MODEL_PATH --peft_model_name $PEFT_MODEL_PATH --tp_size 8 --user_prompt "Hello my name is"
```
The script will ask for another prompt ina loop after completing the generation which you can exit by simply pressing enter and leaving the prompt empty.
When using multiple gpus the model will automatically be split accross the available GPUs using tensor parallelism.

## Multi-node multi-gpu inference
The FP8 quantized variants of Meta Llama (i.e. meta-llama/Meta-Llama-3.1-405B-FP8 and meta-llama/Meta-Llama-3.1-405B-Instruct-FP8) can be executed on a single node with 8x80GB H100 using the script located in this folder.
To run the unquantized Meta Llama 405B variants (i.e. meta-llama/Meta-Llama-3.1-405B and meta-llama/Meta-Llama-3.1-405B-Instruct) we need multi-node inference.
vLLM allows this by leveraging pipeline parallelism accros nodes while still applying tensor parallelism insid each node.
To start a multi-node inference we first need to set up a ray serves which well be leveraged by vLLM to execute the model across node boundaries.

```bash
# On the head node we start the clustr as follows
ray start --head

# After the server starts it prints out a couple of lines including the command to add nodes to the cluster e.g.:
# To add another node to this Ray cluster, run
#   ray start --address='<head-node-ip-address>:6379'
# Where the head node ip address will depend on your environment

# We can then add the worker nodes by executing the command in a shell on the worker node
ray start --address='<head-node-ip-address>:6379'

# We can check if the cluster was launched successfully by executing this on any node
ray status

# It should show the number of nodes we have added as well as the head node
# Node status
# ---------------------------------------------------------------
# Active:
#  1 node_82143b740a25228c24dc8bb3a280b328910b2fcb1987eee52efb838b
#  1 node_3f2c673530de5de86f953771538f35437ab60e3cacd7730dbca41719
```

To launch the inference we can then execute the inference script while we adapt pp_size and tp_size to our environment.

```
pp_size - number of worker + head nodes

tp_size - number of GPUs per node
```

If our environment consists of two nodes with 8 GPUs each we would execute:
```bash
python inference.py --model_name $MODEL_PATH --peft_model_name $PEFT_MODEL_PATH --pp_size 2 --tp_size 8 --user_prompt "Hello my name is"
```

The launch of the vLLM engine will take some time depending on your environment as each worker will need to load the checkpoint files to extract its fraction of the weights.
and even if it seem to hang