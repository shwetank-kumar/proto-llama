from datasets import load_dataset
from tqdm import tqdm
import torch
import sqlite3
from datetime import datetime
import json
from typing import Dict, List, Any, Optional
from accelerate import Accelerator
from pathlib import Path
import logging
from outlines import models, generate, samplers
from transformers import AutoModelForCausalLM, AutoTokenizer
from mmlu_pro_constrained.config import Answer, ModelConfig, DEFAULT_CONFIG

class MMLPROEvaluatorConstr:
    def __init__(
        self,
        model_name: str,
        tokenizer,
        config,
        accelerator: Optional[Accelerator] = None,
        output_dir: str = "mmlu_pro_constrained",
        db_name: str = "mmlu_eval_results.db"
    ):
        """
        Initialize the MMLU-Pro evaluator with constrained generation.
        
        Args:
            model_name: Name of the model to load
            tokenizer: Pre-initialized tokenizer
            config: ModelConfig instance containing evaluation parameters
            accelerator: Optional Accelerator instance for distributed training
        """
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.config = config
        self.accelerator = accelerator or Accelerator(
            mixed_precision='bf16',
            gradient_accumulation_steps=1,
            split_batches=True
        )
        
        # Load model with appropriate configuration
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=getattr(torch, self.config.torch_dtype),
            device_map="auto"
        )
        
        # Initialize Outlines model and generator
        self.llm = models.Transformers(self.model, self.tokenizer)
        self.sampler = samplers.multinomial(
            temperature=self.config.temperature if self.config.temperature is not None else 1.0,
            top_p=self.config.top_p if self.config.top_p is not None else 1.0
        )
        self.generator = generate.json(self.llm, Answer, self.sampler)
        
        # Create output directory if it doesn't exist
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.output_dir / db_name
        
        # Initialize database connection
        self.db_conn = self._init_db()

    def _init_db(self) -> sqlite3.Connection:
        """Initialize SQLite database for storing evaluation results."""
        conn = sqlite3.connect(self.db_path)
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
        ''', (self.model_name, self.config.subject))
        
        current_max = c.fetchone()[0]
        eval_index = 1 if current_max is None else current_max + 1
        
        generation_params = {
            'temperature': self.config.temperature,
            'top_p': self.config.top_p,
            'constrained_generation': True
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
            self.model_name,
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
        Evaluate the model on the given dataset using constrained generation.
        Skips batches that cause errors during generation.
        """
        import logging  # Add import at top of file
        
        self.model.eval()
        dataset = list(dataset)
        
        predictions = []
        references = []
        reasonings = []
        skipped_indices = []

        for i in tqdm(range(0, len(dataset), self.config.batch_size), desc="Processing batches"):
            batch = dataset[i:i + self.config.batch_size]
            batch_prompts = []
        
            # Prepare prompts
            for example in batch:
                options_text = "\n".join(f"{chr(65+idx)}) {option}" 
                                    for idx, option in enumerate(example['options']))
                prompt = self.config.prompt_template.format(
                    question=example['question'],
                    options=options_text
                )
                batch_prompts.append(prompt)
                
                if self.config.verbose:
                    print(f"\nPrompt:")
                    print(prompt)
            
            # Generate answers with constrained generation
            if self.config.verbose:
                print("\nGenerating responses...")
                
            try:
                with torch.inference_mode(), torch.amp.autocast(device_type='cuda'):
                    answers = self.generator(
                        batch_prompts, 
                        max_tokens=self.config.max_new_tokens
                    )

                if self.config.verbose:
                    print("\nResults:")
                    print("-" * 50)
                
                for example, answer in zip(batch, answers):
                    pred = ord(answer.choice) - 65
                    predictions.append(pred)
                    references.append(example['answer_index'])
                    reasonings.append(answer.reasoning)
                    
                    if self.config.verbose:
                        print(f"\nReasoning: {answer.reasoning}")
                        print(f"Predicted answer: {answer.choice}")
                        print(f"Correct answer: {chr(65 + example['answer_index'])}")
                        print(f"Correct? {'✓' if pred == example['answer_index'] else '✗'}")
                        
            except Exception as e:
                # Log the error and batch index
                logging.error(f"Error processing batch starting at index {i}: {str(e)}")
                # Store the indices of skipped examples
                skipped_indices.extend(range(i, min(i + self.config.batch_size, len(dataset))))
                continue

        # Calculate accuracy only on processed examples
        if predictions:
            accuracy = sum(p == r for p, r in zip(predictions, references)) / len(predictions)
        else:
            accuracy = 0.0
            
        results = {
            'predictions': predictions,
            'references': references,
            'reasonings': reasonings,
            'accuracy': accuracy,
            'skipped_indices': skipped_indices,
            'processed_examples': len(predictions),
            'total_examples': len(dataset)
        }
        
        # Log summary statistics
        logging.info(f"Evaluation completed. Processed {len(predictions)}/{len(dataset)} examples")
        if skipped_indices:
            logging.info(f"Skipped {len(skipped_indices)} examples at indices: {skipped_indices}")
        
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