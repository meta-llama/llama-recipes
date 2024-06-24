# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass
from typing import Optional
import torch
from transformers import BitsAndBytesConfig

@dataclass
class quantizatio_config:
    quant_type: str  # "int4" or "int8"
    compute_dtype: torch.dtype
    use_double_quant: bool = False
    quant_storage: torch.dtype = torch.bfloat16

    def create_bnb_config(self) -> BitsAndBytesConfig:
        if self.quant_type not in {"int4", "int8"}:
            raise ValueError("quant_type must be either 'int4' or 'int8'")

        config_params = {
            "bnb_4bit_quant_type" if self.quant_type == "int4" else "bnb_8bit_quant_type": self.quant_type,
            "bnb_4bit_compute_dtype" if self.quant_type == "int4" else "bnb_8bit_compute_dtype": self.compute_dtype,
            "bnb_4bit_use_double_quant" if self.quant_type == "int4" else "bnb_8bit_use_double_quant": self.use_double_quant,
            "bnb_4bit_quant_storage" if self.quant_type == "int4" else "bnb_8bit_quant_storage": self.quant_storage,
        }

        if self.quant_type == "int4":
            return BitsAndBytesConfig(load_in_4bit=True, **config_params)
        else:
            return BitsAndBytesConfig(load_in_8bit=True, **config_params)
