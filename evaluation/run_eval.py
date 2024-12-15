import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from datasets import load_dataset
from mmlu_pro.eval import MMPROEvaluator
from mmlu_pro.config import ModelConfig, DEFAULT_CONFIG

def main():
    # Set random seeds
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    # Initialize config (you can modify this as needed)
    config = DEFAULT_CONFIG
    
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision='bf16',
        gradient_accumulation_steps=1,
        split_batches=True
    )

    # Initialize model and tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
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

    # Prepare model with accelerator
    model = accelerator.prepare(model)

    # Initialize evaluator
    evaluator = MMPROEvaluator(model, tokenizer, config, accelerator)

    # Load and filter dataset
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    subject_data = dataset.filter(lambda x: x['category'] == config.subject)
    
    if len(subject_data) == 0:
        available_subjects = evaluator.get_available_subjects()
        print(f"Error: Subject '{config.subject}' not found.")
        print("Available subjects:", available_subjects)
        return

    print(f"Evaluating {config.model_name} on subject: {config.subject}")
    print(f"Using prompt template: {config.prompt_id}")
    print(f"Number of BOS tokens: {config.n_bos_tokens}")
    print("-" * 50)

    # Run evaluation
    results = evaluator.evaluate(subject_data)
    print(f"Accuracy: {results['accuracy']:.4f}")
    
    # Clean up
    evaluator.close()

if __name__ == "__main__":
    main()