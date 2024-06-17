from dataclasses import dataclass, field
from typing import List


@dataclass
class QuantizationConfig:
    quant_type: str  # "int4" or "int8"
    compute_dtype: torch.dtype
    use_double_quant: bool = False
    quant_storage: torch.dtype = torch.bfloat16

    def create_bnb_config(self):
        if self.quant_type == "int4":
            quant_type_str = "nf4"
        elif self.quant_type == "int8":
            quant_type_str = "int8"
        else:
            raise ValueError("quant_type must be either 'int4' or 'int8'")

        return BitsAndBytesConfig(
            load_in_4bit=(self.quant_type == "int4"),
            bnb_4bit_quant_type=quant_type_str,
            bnb_4bit_compute_dtype=self.compute_dtype,
            bnb_4bit_use_double_quant=self.use_double_quant,
            bnb_4bit_quant_storage=self.quant_storage
        )
