import torch
import argparse
from transformers import AutoTokenizer
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
from llmcompressor.transformers.compression.helpers import (  # noqa
    calculate_offload_device_map,
    custom_offload_device_map,
)

def main():
    parser = argparse.ArgumentParser(description="Compress a language model.")
    parser.add_argument("model_stub", type=str, help="The model stub (e.g., 'bosonai/Higgs-Llama-3-70B')")
    args = parser.parse_args()

    recipe = """
    quant_stage:
        quant_modifiers:
            QuantizationModifier:
                ignore: ["lm_head"]
                config_groups:
                    group_0:
                        weights:
                            num_bits: 8
                            type: float
                            strategy: channel
                            dynamic: false
                            symmetric: true
                        input_activations:
                            num_bits: 8
                            type: float
                            strategy: token
                            dynamic: true
                            symmetric: true
                        targets: ["Linear"]
    """

    model_stub = args.model_stub
    model_name = model_stub.split("/")[-1]

    device_map = calculate_offload_device_map(
        model_stub, reserve_for_hessians=False, num_gpus=1, torch_dtype=torch.float16
    )

    model = SparseAutoModelForCausalLM.from_pretrained(
        model_stub, torch_dtype=torch.float16, device_map=device_map
    )

    output_dir = f"./{model_name}-FP8-dynamic"

    oneshot(
        model=model,
        recipe=recipe,
        output_dir=output_dir,
        save_compressed=True,
        tokenizer=AutoTokenizer.from_pretrained(model_stub),
    )

if __name__ == "__main__":
    main()