from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    subject: str = "computer science"
    batch_size: int = 8
    max_new_tokens: int = 8192
    do_sample: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"

# Can be imported and used directly for default config
DEFAULT_CONFIG = ModelConfig()