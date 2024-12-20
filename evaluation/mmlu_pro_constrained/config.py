from dataclasses import dataclass
from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field

# Predefined prompt templates
PROMPT_TEMPLATES = {
    "basic": "Question: {question}\nOptions:\n{options}\nAnswer: Let's solve this step by step.",
    "json_basic": "Reason the following question step by step and provide the answer choice and reasoning.\n\nQuestion: {question}\nOptions:\n{options}",
    "systematic": """You are a helpful AI taking a multiple choice exam. For each question:
1. Read the question carefully
2. Consider each option systematically
3. Explain your reasoning
4. Select the final answer as a single letter [A-J]

Question: {question}

Options:
{options}""",
    "concise": "Question: {question}\nOptions:\n{options}\nAnswer:",
    "default": "The following are multiple choice questions (with answers) about {subject}. Think step by step before selecting your answer as a single letter [A-J]. Question: {question}\nOptions:\n{options}\nAnswer:",
}

MAX_TOKENS = 512

class AnswerChoice(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"
    H = "H"
    I = "I"
    J = "J"

class Answer(BaseModel):
    """Schema for constrained generation output using Outlines"""
    reasoning: str = Field(..., max_length=MAX_TOKENS, description="Step-by-step reasoning for the answer")
    choice: AnswerChoice = Field(
        ...,
        description="The chosen answer option",
        examples=["A", "B"]
    )

@dataclass
class ModelConfig:
    # Model configuration
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    subject: str = "computer science"
    
    # Generation settings
    batch_size: int = 16
    max_new_tokens: int = MAX_TOKENS
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 0.1
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    
    # Prompt configuration
    prompt_id: str = "json_basic"
    verbose: bool = False

    @property
    def prompt_template(self):
        return PROMPT_TEMPLATES.get(self.prompt_id, PROMPT_TEMPLATES["basic"])

# Default configuration
DEFAULT_CONFIG = ModelConfig()