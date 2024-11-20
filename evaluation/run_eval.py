from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from lm_eval.models import huggingface
from lm_eval.tasks import get_task_dict
from lm_eval import evaluate
from lm_eval.utils import simple_parse_args_string
import argparse
import sqlite3
from datetime import datetime

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def setup_database():
    conn = sqlite3.connect('model_evaluations.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT,
            task_name TEXT,
            temperature REAL,
            do_sample BOOLEAN,
            top_p REAL,
            batch_size INTEGER,
            max_length INTEGER,
            device TEXT,
            score REAL,
            timestamp DATETIME,
            additional_metrics TEXT
        )
    ''')
    conn.commit()
    return conn

def run_evaluation(args):
    print("\nDebug - Input args:")
    print(f"temperature: {args.temperature}")
    print(f"do_sample: {args.do_sample}")
    print(f"top_p: {args.top_p}")

    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Create generation kwargs dictionary based on do_sample
    generation_kwargs = {
        "do_sample": args.do_sample,
        "max_length": args.max_length,
    }
    
    # Only add temperature and top_p if do_sample is True
    if args.do_sample:
        generation_kwargs.update({
            "temperature": args.temperature,
            "top_p": args.top_p,
        })

    print("\nDebug - Generation kwargs:", generation_kwargs)

    model_wrapper = huggingface.HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        generation_kwargs=generation_kwargs
    )

    # Get task dict
    task_dict = get_task_dict([args.task_name])

    results = evaluate(
        lm=model_wrapper,
        task_dict=task_dict
    )

    return results, accelerator.device

def save_results(conn, args, results, device):
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO evaluations (
            model_id, task_name, temperature, do_sample, top_p,
            batch_size, max_length, device, score, timestamp, additional_metrics
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        args.model_id,
        args.task_name,
        args.temperature,
        args.do_sample,
        args.top_p,
        args.batch_size,
        args.max_length,
        str(device),
        results['results'][f"{args.task_name}_acc"],
        datetime.now(),
        str(results)
    ))
    conn.commit()

def main():
    parser = argparse.ArgumentParser(description='Run model evaluation')
    parser.add_argument('--model_id', type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                      help='Model identifier')
    parser.add_argument('--task_name', type=str, default="mmlu_pro_computer_science",
                      help='Task name for evaluation')
    # Changed bool type to str2bool for proper boolean parsing
    parser.add_argument('--do_sample', type=str2bool, default=True,
                      help='Whether to use sampling (true/false)')
    parser.add_argument('--temperature', type=float, default=0.7,
                      help='Sampling temperature (only used if do_sample=True)')
    parser.add_argument('--top_p', type=float, default=0.9,
                      help='Top-p sampling parameter (only used if do_sample=True)')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size')
    parser.add_argument('--max_length', type=int, default=256,
                      help='Maximum sequence length')

    args = parser.parse_args()
    conn = setup_database()

    try:
        results, device = run_evaluation(args)
        save_results(conn, args, results, device)
        print(f"Evaluation completed and saved. Score: {results['results'][f'{args.task_name}_acc']}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()