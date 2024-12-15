import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
import sqlite3
from datetime import datetime
import json
import re
import random
import argparse
from torch.utils.data import DataLoader
from evaluation.mmlu_pro_constrained_draft.constrained_gen_config_draft import ModelConfig, DEFAULT_CONFIG, PROMPT_TEMPLATES
from formatron.integrations.transformers import create_formatter_logits_processor_list
from formatron.formatter import FormatterBuilder

def evaluate_model(model, tokenizer, dataset, config: ModelConfig):
    model.eval()
    dataset = list(dataset)
    
    f = FormatterBuilder()
    schema = config.grammar_schema
    f.append_line(f"{f.json(schema, capture_name='json')}")
    
    # Process outputs
    predictions = []
    references = []
    
    # dataset = dataset[0:config.batch_size]

    for i in tqdm(range(0, len(dataset), config.batch_size), desc="Processing batches"):
        batch = dataset[i:i + config.batch_size]
        batch_inputs = []
    
        # Prepare prompts and encode with BOS tokens
        for example in batch:
            options_text = "\n".join(f"{chr(65+idx)}) {option}" 
                                for idx, option in enumerate(example['options']))
            prompt = f"""<|system|>You are a helpful AI taking a multiple choice exam.<|end|><|user|>{config.prompt_template.format(question=example['question'], options=options_text)}<|end|><|assistant|>"""
            
            # Encode prompt and add BOS tokens
            bos_tokens = [tokenizer.bos_token_id] * config.n_bos_tokens
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            batch_inputs.append(bos_tokens + input_ids)
            
            # Print each prompt only if verbose is True
            if config.verbose:
                print(f"\nPrompt {len(batch_inputs)}:")
                print(prompt)
        
        # Pad sequences to same length
        max_len = max(len(ids) for ids in batch_inputs)
        padded_inputs = [
            [tokenizer.pad_token_id] * (max_len - len(ids)) + ids 
            for ids in batch_inputs
        ]
        
        # Convert to tensor and move to device
        input_tensor = torch.tensor(padded_inputs).to(accelerator.device)
        logits_processor = create_formatter_logits_processor_list(tokenizer, len(batch)*[f])
        # Generate with mixed precision
        if config.verbose:
            print("\nGenerating responses...")
        with torch.inference_mode(), torch.amp.autocast(device_type='cuda'):
            outputs = model.generate(
                input_tensor,
                max_new_tokens=config.max_new_tokens,
                do_sample=config.do_sample,
                temperature=config.temperature,
                top_p=config.top_p,
                pad_token_id=tokenizer.pad_token_id,
                logits_processor=logits_processor
            )
        
        # Decode only the new tokens for each sequence
        decoded_batch = []
        results = []
        for output, input_ids, lpc in zip(outputs, padded_inputs, logits_processor):
            generated_text = tokenizer.batch_decode(output[len(input_ids):], skip_special_tokens=True)
            decoded_batch.extend(generated_text)
            results.extend(lpc.formatters_captures)
        
        if config.verbose:
            print("\nResults:")
            print("-" * 50)
            
        print(results)
            
        predictions.extend(
            ord(r['json'].dict()['answer'].upper()) - 65 if ('json' in r and r['json'].dict()['answer'] is not None)
            else random.randint(0, 9) 
            for r in results
        )
        references.extend([b['answer_index'] for b in batch])
    
    print(predictions, references)
    accuracy = sum(p == r for p, r in zip(predictions, references)) / len(predictions)
    print(f"Accuracy: {accuracy:.2%}")
    
    return {
        'predictions': predictions,
        'references': references,
        'accuracy': accuracy
    }
        
def init_db():
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
    parser.add_argument('--verbose', action='store_true', 
                       help='Print detailed prompts and results')

    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True  # Optimize CUDA operations

    if args.list_subjects:
        subjects = get_available_subjects()
        print("Available subjects:", subjects)
        return

    config = DEFAULT_CONFIG

    if args.prompt:
        config.prompt_id = args.prompt
    elif args.custom_prompt:
        PROMPT_TEMPLATES['custom'] = args.custom_prompt
        config.prompt_id = 'custom'

    if args.subject:
        config.subject = args.subject
    if args.model:
        config.model_name = args.model
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.verbose:
        config.verbose = True

    global accelerator
    accelerator = Accelerator(
        mixed_precision='bf16',
        gradient_accumulation_steps=1,
        split_batches=True
    )
    conn = init_db()

    print(f"Evaluating {config.model_name} on subject: {config.subject}")
    print(f"Using prompt template: {config.prompt_id}")
    print("Config:", config)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=getattr(torch, config.torch_dtype),
        device_map=config.device_map
    )
    model.config.pad_token_id = model.config.bos_token_id
    tokenizer.pad_token = tokenizer.bos_token
    model.resize_token_embeddings(len(tokenizer))

    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    subject_data = dataset.filter(lambda x: x['category'] == config.subject)
    
    dataset = accelerator.prepare(subject_data)
    model = accelerator.prepare(model)
    
    if len(subject_data) == 0:
        available_subjects = get_available_subjects()
        print(f"Error: Subject '{config.subject}' not found.")
        print("Available subjects:", available_subjects)
        return
    
    print(f"Number of BOS tokens: {config.n_bos_tokens}")
    print("-" * 50)

    results = evaluate_model(model, tokenizer, subject_data, config)
    store_results(conn, results, config)
    
    print(f"Accuracy: {results['accuracy']:.4f}")
    
    conn.close()

if __name__ == "__main__":
    main()