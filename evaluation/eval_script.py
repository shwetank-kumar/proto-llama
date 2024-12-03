# from accelerate import Accelerator
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from datasets import load_dataset
# import json
# import argparse
# from datetime import datetime
# import logging
# import numpy as np
# from typing import List, Dict, Any

# def setup_logging():
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s'
#     )

# def create_model_wrapper(model_id: str) -> tuple:
#     logging.info(f"Initializing model wrapper for {model_id}")
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
#         tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set
        
#         model = AutoModelForCausalLM.from_pretrained(
#             model_id, 
#             device_map='auto', 
#             torch_dtype=torch.bfloat16,
#             pad_token_id=tokenizer.pad_token_id
#         )
#         return model, tokenizer
#     except Exception as e:
#         logging.error(f"Failed to create model wrapper: {str(e)}")
#         raise

# def evaluate_example(model: AutoModelForCausalLM,
#                     tokenizer: AutoTokenizer,
#                     example: Dict[str, Any],
#                     accelerator: Accelerator) -> str:
#     prompt = f"{example['question']}\nA. {example['A']}\nB. {example['B']}\nC. {example['C']}\nD. {example['D']}\nAnswer:"
    
#     inputs = tokenizer(
#         prompt, 
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=512
#     ).to(accelerator.device)
    
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=1,
#             pad_token_id=tokenizer.pad_token_id,
#             eos_token_id=tokenizer.eos_token_id,
#             do_sample=False  # Use greedy decoding for evaluation
#         )
    
#     predicted = tokenizer.decode(outputs[0][-1:], skip_special_tokens=True).strip()
#     # Map the output to ABCD if it's not already in that format
#     predicted = predicted.upper()
#     if predicted in ['A', 'B', 'C', 'D']:
#         return predicted
#     else:
#         logging.warning(f"Invalid prediction '{predicted}', defaulting to 'A'")
#         return 'A'

# def run_evaluation(model: AutoModelForCausalLM, 
#                   tokenizer: AutoTokenizer, 
#                   tasks: List[str], 
#                   accelerator: Accelerator) -> Dict[str, Any]:
#     all_results = {}
    
#     for task in tasks:
#         logging.info(f"Starting evaluation for task: {task}")
#         try:
#             # Load the correct dataset
#             dataset = load_dataset("TIGER-Lab/MMLU-Pro", task, split="test")
            
#             if dataset is None:
#                 raise ValueError(f"Could not load dataset for task: {task}")
            
#             correct = 0
#             total = 0
#             predictions = []
            
#             for example in dataset:
#                 predicted = evaluate_example(model, tokenizer, example, accelerator)
#                 actual = example['answer']
#                 predictions.append({
#                     'question': example['question'],
#                     'predicted': predicted,
#                     'actual': actual,
#                     'correct': predicted == actual
#                 })
#                 if predicted == actual:
#                     correct += 1
#                 total += 1
            
#             accuracy = correct / total if total > 0 else 0
#             results = {
#                 'accuracy': accuracy,
#                 'total_examples': total,
#                 'correct_predictions': correct,
#                 'predictions': predictions
#             }
            
#             all_results[task] = results
#             logging.info(f"Completed evaluation for {task} with accuracy: {accuracy:.2%}")
            
#         except Exception as e:
#             logging.error(f"Error evaluating task {task}: {str(e)}")
#             all_results[task] = {"error": str(e)}
    
#     return all_results

# def save_results(results: Dict[str, Any], model_id: str, output_file: str = None) -> None:
#     if output_file is None:
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         model_name = model_id.split('/')[-1]
#         output_file = f"eval_results_{model_name}_{timestamp}.json"
    
#     def convert_to_serializable(obj: Any) -> Any:
#         if isinstance(obj, (np.int8, np.int16, np.int32, np.int64,
#                           np.uint8, np.uint16, np.uint32, np.uint64)):
#             return int(obj)
#         elif isinstance(obj, (np.float16, np.float32, np.float64)):
#             return float(obj)
#         elif isinstance(obj, np.ndarray):
#             return obj.tolist()
#         elif isinstance(obj, dict):
#             return {k: convert_to_serializable(v) for k, v in obj.items()}
#         elif isinstance(obj, list):
#             return [convert_to_serializable(i) for i in obj]
#         elif isinstance(obj, (bool, int, float, str)):
#             return obj
#         elif obj is None:
#             return None
#         return str(obj)
    
#     serializable_results = convert_to_serializable(results)
    
#     output_data = {
#         "model_id": model_id,
#         "timestamp": datetime.now().isoformat(),
#         "results": serializable_results
#     }
    
#     try:
#         with open(output_file, 'w') as f:
#             json.dump(output_data, f, indent=2)
#         logging.info(f"Results saved to {output_file}")
#     except Exception as e:
#         logging.error(f"Failed to save results: {str(e)}")
#         raise

# def main():
#     parser = argparse.ArgumentParser(description="Evaluate language models on specified MMLU tasks")
#     parser.add_argument("--model-id", required=True, help="HuggingFace model ID")
#     parser.add_argument("--tasks", nargs="+", required=True, help="List of MMLU tasks to evaluate")
#     parser.add_argument("--output-file", help="Output file path (optional)")
#     args = parser.parse_args()

#     setup_logging()
    
#     try:
#         accelerator = Accelerator()
#         model, tokenizer = create_model_wrapper(args.model_id)
#         model, tokenizer = accelerator.prepare(model, tokenizer)
#         results = run_evaluation(model, tokenizer, args.tasks, accelerator)
#         save_results(results, args.model_id, args.output_file)
        
#     except Exception as e:
#         logging.error(f"Evaluation failed: {str(e)}")
#         raise

# if __name__ == "__main__":
#     main()