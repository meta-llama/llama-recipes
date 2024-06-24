from dataclasses import dataclass, field
from typing import List
import torch
from transformers import BitsAndBytesConfig

@dataclass
class quantizatio_config:
    quant_type: str  # "int4" or "int8"
    compute_dtype: torch.dtype
    use_double_quant: bool = False
    quant_storage: torch.dtype = torch.bfloat16

    def create_bnb_config(self):
        if self.quant_type == "int4":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.compute_dtype,
                bnb_4bit_use_double_quant=self.use_double_quant,
                bnb_4bit_quant_storage=self.quant_storage
            )
        elif self.quant_type == "int8":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_quant_type="int8",
                bnb_8bit_compute_dtype=self.compute_dtype,
                bnb_8bit_use_double_quant=self.use_double_quant,
                bnb_8bit_quant_storage=self.quant_storage
            )
        else:
            raise ValueError("quant_type must be either 'int4' or 'int8'")
