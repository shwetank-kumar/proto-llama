import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
# import evaluate
import sqlite3
from datetime import datetime
import json
import re
import random
import argparse
from config import ModelConfig, DEFAULT_CONFIG, PROMPT_TEMPLATES

def evaluate_model(model, tokenizer, dataset, config: ModelConfig):
    model.eval()
    predictions = []
    references = []
    dataset = list(dataset)

    for i in tqdm(range(0, len(dataset), config.batch_size)):
        batch = dataset[i:i + config.batch_size]
        batch_prompts = []
        
        for example in batch:
            options_text = "\n".join(f"{chr(65+idx)}) {option}" 
                                   for idx, option in enumerate(example['options']))
            prompt = config.prompt_template.format(
                question=example['question'],
                options=options_text,
                subject=config.subject  # Add subject to the format arguments
            )
            batch_prompts.append(prompt)
        
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            with accelerator.autocast():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=config.do_sample,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    pad_token_id=tokenizer.pad_token_id
                )
        
        responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        for j, response in enumerate(responses):
            answer = None
            for pattern in [
                r'.*answer is (?:\()?([A-J])(?:\))?.*',
                r'\.*[aA]nswer:\s*\(([A-J])\)'
            ]:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    answer = match.group(1)
                    break
            
            if answer is None:
                answer = random.choice([chr(65+i) for i in range(len(batch[j]['options']))])
            
            pred = ord(answer) - 65
            predictions.append(pred)
            references.append(batch[j]['answer_index'])

    accuracy = sum(p == r for p, r in zip(predictions, references)) / len(predictions)
    
    return {
        'predictions': predictions,
        'references': references,
        'accuracy': accuracy
    }

# def init_db():
#     conn = sqlite3.connect('mmlu_eval_results.db')
#     c = conn.cursor()
    
#     c.execute('''
#         CREATE TABLE IF NOT EXISTS evaluations (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             model_name TEXT,
#             subject TEXT,
#             generation_params TEXT,  -- JSON object
#             max_tokens INTEGER,
#             torch_dtype TEXT,
#             metrics TEXT,  -- JSON object
#             timestamp TEXT,
#             eval_index INTEGER
#         )
#     ''')
#     conn.commit()
#     return conn

def init_db():
    conn = sqlite3.connect('mmlu_eval_results.db')
    c = conn.cursor()
    
    # First create table if it doesn't exist
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
            eval_index INTEGER
        )
    ''')
    
    # Check if columns exist and add them if they don't
    cursor = c.execute('PRAGMA table_info(evaluations)')
    columns = [row[1] for row in cursor.fetchall()]
    
    if 'prompt_id' not in columns:
        c.execute('ALTER TABLE evaluations ADD COLUMN prompt_id TEXT')
    if 'prompt_template' not in columns:
        c.execute('ALTER TABLE evaluations ADD COLUMN prompt_template TEXT')
    
    conn.commit()
    return conn

def store_results(conn, results, config: ModelConfig):
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    
    c.execute('''
        SELECT MAX(eval_index)
        FROM evaluations 
        WHERE model_name = ? AND subject = ?
    ''', (config.model_name, config.subject))
    
    current_max = c.fetchone()[0]
    eval_index = 1 if current_max is None else current_max + 1
    
    generation_params = {
        'temperature': config.temperature,
        'top_p': config.top_p,
        'do_sample': config.do_sample
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
        config.model_name,
        config.subject,
        json.dumps(generation_params),
        config.max_new_tokens,
        config.torch_dtype,
        json.dumps(metrics),
        timestamp,
        eval_index,
        config.prompt_id,
        config.prompt_template
    ))
    conn.commit()

def get_available_subjects():
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    return list(set(dataset['category']))

def main():
    parser = argparse.ArgumentParser(description='Run MMLU-Pro evaluation')
    parser.add_argument('--subject', type=str, help='Override subject')
    parser.add_argument('--model', type=str, help='Override model name')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--list-subjects', action='store_true', help='List available subjects')
    parser.add_argument('--prompt', type=str, choices=list(PROMPT_TEMPLATES.keys()), 
                       help='Use a predefined prompt template')
    parser.add_argument('--custom-prompt', type=str, 
                       help='Custom prompt template with {question} and {options} placeholders')

    args = parser.parse_args()
    seed = 42  # or add as an argument like --seed
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if args.list_subjects:
        subjects = get_available_subjects()
        print("Available subjects:", subjects)
        return

    config = DEFAULT_CONFIG

    if args.prompt:
        config.prompt_id = args.prompt
    elif args.custom_prompt:
        # Add the custom template to PROMPT_TEMPLATES
        PROMPT_TEMPLATES['custom'] = args.custom_prompt
        config.prompt_id = 'custom'

    if args.subject:
        config.subject = args.subject
    if args.model:
        config.model_name = args.model
    if args.batch_size:
        config.batch_size = args.batch_size

    global accelerator
    accelerator = Accelerator()
    conn = init_db()

    print(f"Evaluating {config.model_name} on subject: {config.subject}")
    print(f"Using prompt template: {config.prompt_id}")
    print("Config:", json.dumps(config.__dict__, indent=2))

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=getattr(torch, config.torch_dtype),
        device_map=config.device_map
    )
    model.resize_token_embeddings(len(tokenizer))

    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    subject_data = dataset.filter(lambda x: x['category'] == config.subject)

    if len(subject_data) == 0:
        available_subjects = get_available_subjects()
        print(f"Error: Subject '{config.subject}' not found.")
        print("Available subjects:", available_subjects)
        return

    results = evaluate_model(model, tokenizer, subject_data, config)
    store_results(conn, results, config)
    
    print(f"Accuracy: {results['accuracy']:.4f}")
    
    conn.close()

if __name__ == "__main__":
    main()