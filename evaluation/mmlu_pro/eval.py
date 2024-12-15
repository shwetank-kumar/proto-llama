from datasets import load_dataset
from tqdm import tqdm
import torch
import sqlite3
from datetime import datetime
import json
import re
import random
# from dataclasses import asdict
from typing import Dict, List, Any, Optional
from accelerate import Accelerator

class MMPROEvaluator:
    def __init__(
        self,
        model,
        tokenizer,
        config,
        accelerator: Optional[Accelerator] = None
    ):
        """
        Initialize the MMLU-Pro evaluator with a pre-initialized model and tokenizer.
        
        Args:
            model: Pre-initialized model
            tokenizer: Pre-initialized tokenizer
            config: ModelConfig instance containing evaluation parameters
            accelerator: Optional Accelerator instance for distributed training
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.accelerator = accelerator or Accelerator(
            mixed_precision='bf16',
            gradient_accumulation_steps=1,
            split_batches=True
        )
        
        # Pre-compile regex patterns for efficiency
        self.answer_patterns = [
            re.compile(r'.*answer is (?:\()?([A-J])(?:\))?.*', re.IGNORECASE),
            re.compile(r'\.*[aA]nswer:\s*\(([A-J])\)', re.IGNORECASE),
            re.compile(r'.*\b([A-J])\b.*', re.IGNORECASE)
        ]
        
        # Initialize database connection
        self.db_conn = self._init_db()

    def _init_db(self) -> sqlite3.Connection:
        """Initialize SQLite database for storing evaluation results."""
        conn = sqlite3.connect('mmlu_eval_results.db')
        c = conn.cursor()
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                subject TEXT,
                generation_params TEXT,
                max_tokens INTEGER,
                torch_dtype TEXT,
                metrics TEXT,
                timestamp TEXT,
                eval_index INTEGER,
                prompt_id TEXT,
                prompt_template TEXT
            )
        ''')
        
        conn.commit()
        return conn

    def _store_results(self, results: Dict[str, Any]) -> None:
        """Store evaluation results in the database."""
        c = self.db_conn.cursor()
        timestamp = datetime.now().isoformat()
        
        c.execute('''
            SELECT MAX(eval_index)
            FROM evaluations 
            WHERE model_name = ? AND subject = ?
        ''', (self.config.model_name, self.config.subject))
        
        current_max = c.fetchone()[0]
        eval_index = 1 if current_max is None else current_max + 1
        
        generation_params = {
            'temperature': self.config.temperature,
            'top_p': self.config.top_p,
            'do_sample': self.config.do_sample
        }
        
        metrics = {
            'accuracy': results['accuracy']
        }
        
        c.execute('''
            INSERT INTO evaluations 
            (model_name, subject, generation_params, max_tokens, torch_dtype, 
             metrics, timestamp, eval_index, prompt_id, prompt_template)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            self.config.model_name,
            self.config.subject,
            json.dumps(generation_params),
            self.config.max_new_tokens,
            self.config.torch_dtype,
            json.dumps(metrics),
            timestamp,
            eval_index,
            self.config.prompt_id,
            self.config.prompt_template
        ))
        self.db_conn.commit()

    def evaluate(self, dataset) -> Dict[str, Any]:
        """
        Evaluate the model on the given dataset.
        
        Args:
            dataset: Dataset to evaluate on
            
        Returns:
            Dictionary containing evaluation results
        """
        self.model.eval()
        dataset = list(dataset)
        
        predictions = []
        references = []

        for i in tqdm(range(0, len(dataset), self.config.batch_size), desc="Processing batches"):
            batch = dataset[i:i + self.config.batch_size]
            batch_inputs = []
        
            # Prepare prompts and encode with BOS tokens
            for example in batch:
                options_text = "\n".join(f"{chr(65+idx)}) {option}" 
                                    for idx, option in enumerate(example['options']))
                prompt = self.config.prompt_template.format(
                    question=example['question'],
                    options=options_text
                )
                
                # Encode prompt and add BOS tokens
                bos_tokens = [self.tokenizer.bos_token_id] * self.config.n_bos_tokens
                input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                batch_inputs.append(bos_tokens + input_ids)
                
                if self.config.verbose:
                    print(f"\nPrompt {len(batch_inputs)}:")
                    print(prompt)
            
            # Pad sequences to same length
            max_len = max(len(ids) for ids in batch_inputs)
            padded_inputs = [
                [self.tokenizer.pad_token_id] * (max_len - len(ids)) + ids 
                for ids in batch_inputs
            ]
            
            # Convert to tensor and move to device
            input_tensor = torch.tensor(padded_inputs).to(self.accelerator.device)
            
            # Generate with mixed precision
            if self.config.verbose:
                print("\nGenerating responses...")
            with torch.inference_mode(), torch.amp.autocast(device_type='cuda'):
                outputs = self.model.generate(
                    input_tensor,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature if self.config.temperature is not None else 1.0,
                    top_p=self.config.top_p if self.config.top_p is not None else None,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode only the new tokens for each sequence
            decoded_batch = []
            for output, input_ids in zip(outputs, padded_inputs):
                generated_text = self.tokenizer.decode(output[len(input_ids):], skip_special_tokens=True)
                decoded_batch.append(generated_text)
            
            if self.config.verbose:
                print("\nResults:")
                print("-" * 50)
            
            for i, (example, response) in enumerate(zip(batch, decoded_batch)):
                if self.config.verbose:
                    print(f"\nExample {i+1} response: {response}")
                
                answer = None
                for pattern in self.answer_patterns:
                    if match := pattern.search(response):
                        answer = match.group(1)
                        break
                
                if answer is None:
                    answer = random.choice([chr(65+i) for i in range(len(example['options']))])
                
                pred = ord(answer.upper()) - 65
                predictions.append(pred)
                references.append(example['answer_index'])
                
                if self.config.verbose:
                    print(f"Extracted answer: {answer}")
                    print(f"Correct answer: {chr(65 + example['answer_index'])}")
                    print(f"Correct? {'✓' if pred == example['answer_index'] else '✗'}")
        
        accuracy = sum(p == r for p, r in zip(predictions, references)) / len(predictions)
        
        results = {
            'predictions': predictions,
            'references': references,
            'accuracy': accuracy
        }
        
        self._store_results(results)
        return results

    def close(self):
        """Close the database connection."""
        self.db_conn.close()

    @staticmethod
    def get_available_subjects() -> List[str]:
        """Get list of available subjects in MMLU-Pro dataset."""
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        return list(set(dataset['category']))