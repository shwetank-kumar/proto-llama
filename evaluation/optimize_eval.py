import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from datasets import load_dataset
from mmlu_pro.eval import MMPROEvaluator
from mmlu_pro.config import ModelConfig, DEFAULT_CONFIG
import optuna
from copy import deepcopy
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mmlu_optimization.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MMPROOptimizer:
    def __init__(
        self,
        base_config: ModelConfig,
        n_trials: int = 20,
        output_dir: str = "mmlu_pro",
        db_name: str = "mmlu_pro_optimization.db",
        study_name: Optional[str] = None  # Add optional study name parameter
    ):
        # Create output directory if it doesn't exist
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_path = self.output_dir / "mmlu_pro_optimization.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path, mode='a'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Setup database path
        self.db_path = self.output_dir / db_name
        self.storage_url = f"sqlite:///{self.db_path}"
        
        self.base_config = base_config
        self.base_config.max_new_tokens = 1024
        self.n_trials = n_trials
        
        # Use provided study name or create new one
        self.study_name = study_name if study_name else f"mmlu_opt_{self.base_config.subject}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        try:
            # Check if study exists before creating/loading
            existing_studies = optuna.study.get_all_study_summaries(self.storage_url)
            study_exists = any(s.study_name == self.study_name for s in existing_studies)
            
            if study_exists:
                self.logger.info(f"Resuming existing study: {self.study_name}")
            else:
                if study_name:  # If study name was provided but doesn't exist
                    raise ValueError(f"Study '{study_name}' not found in database")
                self.logger.info(f"Creating new study: {self.study_name}")
            
            self.setup_environment()
            
            self.study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage_url,
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
                load_if_exists=True
            )

            # Log study status
            n_completed = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            n_failed = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL])
            self.logger.info(f"Study has {n_completed} completed and {n_failed} failed trials")
            
        except Exception as e:
            self.logger.error(f"Error during initialization: {str(e)}", exc_info=True)
            raise

    def setup_environment(self):
        """Initialize all necessary components with detailed logging."""
        try:
            logger.info("Setting up random seeds")
            seed = 42
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.benchmark = True
                logger.info("CUDA is available. GPU information:")
                for i in range(torch.cuda.device_count()):
                    logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                    free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
                    logger.info(f"Free memory on GPU {i}: {free_memory / 1024**2:.2f} MB")

            logger.info("Initializing accelerator")
            self.accelerator = Accelerator(
                mixed_precision='bf16',
                gradient_accumulation_steps=1,
                split_batches=True
            )
            logger.info("Accelerator initialized successfully")

            logger.info("Loading tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_config.model_name)
            self.tokenizer.padding_side = "left"
            logger.info("Tokenizer loaded successfully")
            
            logger.info("Loading model")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_config.model_name,
                torch_dtype=getattr(torch, self.base_config.torch_dtype),
                device_map=self.base_config.device_map
            )
            logger.info("Model loaded successfully")
            
            logger.info("Configuring model tokens")
            self.model.config.pad_token_id = self.model.config.bos_token_id
            self.tokenizer.pad_token = self.tokenizer.bos_token
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info("Token configuration completed")

            logger.info("Preparing model with accelerator")
            self.model = self.accelerator.prepare(self.model)
            logger.info("Model preparation completed")

            logger.info("Loading dataset")
            try:
                # Load dataset in streaming mode
                dataset = load_dataset(
                    "TIGER-Lab/MMLU-Pro", 
                    split="test",
                    streaming=True  # Enable streaming mode
                )
                logger.info("Dataset loaded, filtering by subject")
                
                # Filter in batches
                filtered_data = []
                current_batch = []
                batch_size = 100  # Process 100 examples at a time
                
                logger.info("Starting dataset filtering")
                for example in dataset:
                    if example['category'] == self.base_config.subject:
                        current_batch.append(example)
                        
                    if len(current_batch) >= batch_size:
                        filtered_data.extend(current_batch)
                        current_batch = []
                        logger.info(f"Processed batch, current size: {len(filtered_data)}")
                
                # Add remaining examples
                if current_batch:
                    filtered_data.extend(current_batch)
                
                self.subject_data = filtered_data
                logger.info(f"Dataset filtered, size: {len(self.subject_data)}")

            except Exception as e:
                logger.error(f"Error during dataset processing: {str(e)}", exc_info=True)
                raise

            if len(self.subject_data) == 0:
                raise ValueError(f"Subject '{self.base_config.subject}' not found in dataset.")
                
            logger.info("Environment setup completed successfully")
            
        except Exception as e:
            logger.error(f"Error in setup_environment: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error in setup_environment: {str(e)}", exc_info=True)
            raise

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function for optimization."""
        self.logger.info(f"\nStarting trial {trial.number}")
        
        # Check if this trial was already completed
        if len(trial.params) > 0:
            self.logger.info(f"Trial {trial.number} already has parameters: {trial.params}")
            return trial.value if trial.value is not None else float('-inf')
            
        try:
            # Create config for this trial
            trial_config = deepcopy(self.base_config)
            
            # Sample parameters
            trial_config.temperature = trial.suggest_float('temperature', 0.1, 0.9, step=0.1)
            trial_config.top_p = trial.suggest_float('top_p', 0.1, 0.9, step=0.1)
            
            self.logger.info(f"Trial {trial.number} parameters:")
            self.logger.info(f"Temperature: {trial_config.temperature:.1f}")
            self.logger.info(f"Top_p: {trial_config.top_p:.1f}")

            # Store additional trial information
            trial.set_user_attr('model_name', self.base_config.model_name)
            trial.set_user_attr('subject', self.base_config.subject)
            trial.set_user_attr('timestamp', datetime.now().isoformat())
            trial.set_user_attr('max_new_tokens', self.base_config.max_new_tokens)

            # Initialize evaluator with trial config
            evaluator = MMPROEvaluator(
                self.model,
                self.tokenizer,
                trial_config,
                self.accelerator
            )

            try:
                self.logger.info("Running evaluation")
                results = evaluator.evaluate(self.subject_data)
                accuracy = results['accuracy']
                
                # Store additional metrics
                trial.set_user_attr('accuracy', accuracy)
                trial.set_user_attr('n_examples', len(self.subject_data))
                
                self.logger.info(f"Trial {trial.number} completed with accuracy: {accuracy:.4f}")
                
                return accuracy
                
            finally:
                evaluator.close()
                
        except Exception as e:
            self.logger.error(f"Error in trial {trial.number}: {str(e)}", exc_info=True)
            raise

    def optimize(self) -> dict:
        """Run optimization process and return best parameters."""
        try:
            self.logger.info(f"\nStarting/Resuming optimization for subject: {self.base_config.subject}")
            
            # Get trial states
            trials = self.study.trials
            completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
            failed_trials = [t for t in trials if t.state == optuna.trial.TrialState.FAIL]
            
            self.logger.info(f"Found {len(completed_trials)} completed trials")
            self.logger.info(f"Found {len(failed_trials)} failed trials")
            
            # Find the last failed trial's number
            last_failed_number = max([t.number for t in failed_trials]) if failed_trials else -1
            
            # Calculate remaining trials
            remaining_trials = self.n_trials - len(completed_trials)
            
            self.logger.info(f"Last failed trial number: {last_failed_number}")
            self.logger.info(f"Remaining trials to run: {remaining_trials}")
            self.logger.info(f"Model: {self.base_config.model_name}")
            self.logger.info(f"Study name: {self.study_name}")
            self.logger.info("-" * 50)

            if remaining_trials > 0:
                # Create a callback to skip trials we don't want to retry
                def trial_should_retry(trial):
                    # Skip successful trials and trials before the last failure
                    if trial.state == optuna.trial.TrialState.COMPLETE:
                        return False
                    if trial.number < last_failed_number:
                        return False
                    return True
                
                # Get the parameters that were being tried when the failure occurred
                failed_params = None
                if last_failed_number >= 0:
                    failed_trial = [t for t in trials if t.number == last_failed_number][0]
                    failed_params = failed_trial.params
                    self.logger.info(f"Retrying failed parameters: {failed_params}")

                def retry_objective(trial):
                    # If we have failed parameters and this is the retry
                    if failed_params is not None and trial.number == last_failed_number:
                        # Set the same parameters as the failed trial
                        for param_name, param_value in failed_params.items():
                            if param_name == 'temperature':
                                trial.suggest_float('temperature', param_value, param_value)
                            elif param_name == 'top_p':
                                trial.suggest_float('top_p', param_value, param_value)
                        return self.objective(trial)
                    else:
                        return self.objective(trial)

                self.study.optimize(
                    retry_objective, 
                    n_trials=remaining_trials,
                    callbacks=[lambda study, trial: trial_should_retry(trial)]
                )
            else:
                self.logger.info("All trials completed, no additional trials needed")
            
            # Return study results
            results = {
                'best_parameters': self.study.best_params,
                'best_accuracy': self.study.best_value,
                'completed_trials': len(completed_trials) + remaining_trials,
                'model_name': self.base_config.model_name,
                'subject': self.base_config.subject,
                'study_name': self.study_name
            }
            
            self.logger.info("Optimization completed successfully")
            self.logger.info(f"Best parameters: {results['best_parameters']}")
            self.logger.info(f"Best accuracy: {results['best_accuracy']:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error("Error during optimization", exc_info=True)
            raise

    def get_study_statistics(self) -> dict:
        """Get detailed statistics about the optimization study."""
        try:
            logger.info("Calculating study statistics")
            trials = self.study.trials_dataframe()
            
            stats = {
                'number_of_trials': len(trials),
                'best_accuracy': self.study.best_value,
                'best_parameters': self.study.best_params,
                'best_trial_number': self.study.best_trial.number,
                'datetime_start': trials['datetime_start'].min(),
                'datetime_complete': trials['datetime_complete'].max(),
                'parameter_importance': optuna.importance.get_param_importances(self.study)
            }
            
            logger.info("Statistics calculation completed")
            return stats
            
        except Exception as e:
            logger.error("Error calculating study statistics", exc_info=True)
            raise

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run or continue MMLU optimization')
    parser.add_argument('--continue-study', type=str, help='Name of study to continue')
    parser.add_argument('--list-studies', action='store_true', help='List all existing studies')
    args = parser.parse_args()

    output_dir = "mmlu_pro"
    db_name = "mmlu_pro_optimization.db"
    storage_url = f"sqlite:///{Path(output_dir) / db_name}"

    if args.list_studies:
        # List all existing studies
        studies = optuna.study.get_all_study_summaries(storage_url)
        print("\nExisting studies:")
        for study in studies:
            n_trials = study.n_trials
            print(f"- {study.study_name}: {n_trials} trials, best value: {study.best_trial.value if study.best_trial else 'N/A'}")
        return

    try:
        base_config = DEFAULT_CONFIG
        
        # Create optimizer with optional study name
        optimizer = MMPROOptimizer(
            base_config=base_config,
            n_trials=20,
            output_dir=output_dir,
            db_name=db_name,
            study_name=args.continue_study
        )
        
        # Run optimization
        results = optimizer.optimize()
        
        # Print final results
        print("\nOptimization completed!")
        print(f"Best parameters found:")
        print(f"Temperature: {results['best_parameters']['temperature']:.1f}")
        print(f"Top_p: {results['best_parameters']['top_p']:.1f}")
        print(f"Best accuracy: {results['best_accuracy']:.4f}")
        
        # Print study statistics
        stats = optimizer.get_study_statistics()
        print("\nStudy Statistics:")
        print(f"Total trials completed: {results['completed_trials']}")
        if 'datetime_start' in stats and 'datetime_complete' in stats:
            print(f"Study duration: {stats['datetime_complete'] - stats['datetime_start']}")
        print("\nParameter Importance:")
        for param, importance in stats['parameter_importance'].items():
            print(f"{param}: {importance:.3f}")
        print(f"\nResults stored in: {optimizer.storage_url}")
        
    except Exception as e:
        logger.error("Fatal error in main", exc_info=True)
        raise

if __name__ == "__main__":
    main()

