# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass
from typing import Optional
import torch
from transformers import BitsAndBytesConfig

@dataclass
class quantization_config:
    quant_type: str =  "fp4" # "fp4" or "nf4"
    compute_dtype: torch.dtype = torch.bfloat16
    use_double_quant: bool = False
    quant_storage: torch.dtype = torch.bfloat16

    def create_bnb_config(self, quantization: str) -> BitsAndBytesConfig:
        if quantization not in {"4bit", "8bit"}:
            raise ValueError("quantization must be either '4bit' or '8bit'")

        if quantization == "4bit":
            config_params = {
                "bnb_4bit_quant_type": self.quant_type,
                "bnb_4bit_compute_dtype": self.compute_dtype,
                "bnb_4bit_use_double_quant": self.use_double_quant,
                "bnb_4bit_quant_storage": self.quant_storage,
            }
            
            return BitsAndBytesConfig(load_in_4bit=True, **config_params)
        else:
            return BitsAndBytesConfig(load_in_8bit=True)
