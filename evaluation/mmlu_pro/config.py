from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

# Predefined prompt templates
PROMPT_TEMPLATES = {
    "basic": "Question: {question}\nOptions:\n{options}\nAnswer: Let's solve this step by step. ",
    "json_basic": "Reason the following question step by step and provide the answer, probability that the chosen answer is correct, and the reasoning for arriving at the chosen answer as a json object.\n\nQuestion: {question}\nOptions:\n{options}",
    "systematic": """You are a helpful AI taking a multiple choice exam. For each question:
1. Read the question carefully
2. Consider each option systematically
3. Explain your reasoning briefly
4. Select the final answer in the format "The answer is: [A/B/C/D/E/F/G/H/I/J]"

Question: {question}

Options:
{options}""",
    "concise": "Question: {question}\nOptions:\n{options}\n.",
    "default": "The following are multiple choice questions (with answers) about {subject}. Think step by step and then finish your answer with \"the answer is (X)\" where X is the correct letter choice. Question: {question}\nOptions:\n{options}\nAnswer:",
    "reddit_default": "You are a knowledge expert, you are supposed to answer the multi-choice question to derive your final answer as 'The answer is...'. Question: {question}\nOptions:\n{options}\nAnswer:",
    "reddit_improved": "You are a trivia expert who knows everything, you are tasked to answer the following multiple-choice question. Give your final answer in the format of 'The answer is (chosen multiple-choice option)'. Question: {question}\nOptions:\n{options}\nAnswer:",
    "claude": """You are a decisive expert taking a multiple choice test. For each question:

1. Core concept: One clear sentence identifying what the question is testing
2. Critical fact: One key fact that determines the answer
3. Decision path: One simple IF-THEN statement connecting the fact to your chosen answer
4. Commit to your answer with "Therefore: [letter])"

Do not second-guess or mention alternative possibilities after making your decision.
Do not explain why other answers are wrong.
Do not hedge or qualify your answer.

Question: {question}
Options:
{options}"""
}

@dataclass
class ModelConfig:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    subject: str = "computer science"
    batch_size: int = 16
    do_sample: bool = True
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9 # 0.8, 0.7, 0.6, 0.5
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    prompt_id: str = "basic"
    n_bos_tokens: int = 256
    max_new_tokens: int = 512
    verbose: bool = False
    
    @property
    def prompt_template(self):
        return PROMPT_TEMPLATES.get(self.prompt_id, PROMPT_TEMPLATES["basic"])
 
# Add this line at the end of the file
DEFAULT_CONFIG = ModelConfig()