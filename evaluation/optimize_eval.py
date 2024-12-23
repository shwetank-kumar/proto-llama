import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from datasets import load_dataset
from mmlu_pro.eval import MMPROEvaluator
from mmlu_pro_constrained.eval import MMLPROEvaluatorConstr
from mmlu_pro.config import ModelConfig as UnconstrainedConfig
from mmlu_pro_constrained.config import ModelConfig as ConstrainedConfig
from mmlu_pro.config import DEFAULT_CONFIG as UNCONSTRAINED_DEFAULT
from mmlu_pro_constrained.config import DEFAULT_CONFIG as CONSTRAINED_DEFAULT
import optuna
from copy import deepcopy
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimization.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MMPROOptimizer:
    def __init__(
        self,
        base_config: Union[UnconstrainedConfig, ConstrainedConfig],
        eval_type: str = "unconstrained",
        n_trials: int = 20,
        study_name: Optional[str] = None
    ):
        self.eval_type = eval_type
        if eval_type not in ["constrained", "unconstrained"]:
            raise ValueError("eval_type must be either 'constrained' or 'unconstrained'")

        # Setup appropriate paths based on eval_type
        self.output_dir = Path("mmlu_pro_constrained" if eval_type == "constrained" else "mmlu_pro")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_name = "mmlu_optimization.log"
        log_path = self.output_dir / log_name
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(str(log_path), mode='a'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Setup database path
        db_name = "mmlu_optimization.db"
        self.db_path = self.output_dir / db_name
        self.storage_url = f"sqlite:///{str(self.db_path)}"
        
        self.base_config = base_config
        # self.base_config.max_new_tokens = 16
        self.n_trials = n_trials
        
        # Update study name to include evaluation type
        self.study_name = study_name if study_name else f"mmlu_opt_{eval_type}_{self.base_config.subject}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        try:
            # Check if study exists before creating/loading
            existing_studies = optuna.study.get_all_study_summaries(self.storage_url)
            study_exists = any(s.study_name == self.study_name for s in existing_studies)
            
            if study_exists:
                self.logger.info(f"Resuming existing study: {self.study_name}")
            else:
                if study_name:
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
            
            # Store evaluation type in study user attributes
            if not self.study.user_attrs:
                self.study.set_user_attr('eval_type', eval_type)
            elif self.study.user_attrs.get('eval_type') != eval_type:
                raise ValueError(f"Existing study uses {self.study.user_attrs['eval_type']} evaluation, but {eval_type} was requested")

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
            
            if self.eval_type == "unconstrained":
                logger.info("Loading model for unconstrained evaluation")
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
            dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
            if self.base_config.subject is not None:
                self.subject_data = dataset.filter(lambda x: x['category'] == self.base_config.subject)
            else:
                self.subject_data = dataset

            if len(self.subject_data) == 0:
                raise ValueError(f"Subject '{self.base_config.subject}' not found in dataset.")
                
            logger.info(f"Dataset loaded successfully with {len(self.subject_data)} examples")
            logger.info("Environment setup completed successfully")
            
        except Exception as e:
            logger.error(f"Error in setup_environment: {str(e)}", exc_info=True)
            raise

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function for optimization."""
        self.logger.info(f"\nStarting trial {trial.number}")
        
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
            trial.set_user_attr('eval_type', self.eval_type)

            # Initialize appropriate evaluator
            if self.eval_type == "constrained":
                evaluator = MMLPROEvaluatorConstr(
                    self.base_config.model_name,
                    self.tokenizer,
                    trial_config,
                    self.accelerator
                )
            else:
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
                
                # Store accuracy and other metrics in trial
                trial.set_user_attr('accuracy', float(accuracy))  # Ensure accuracy is stored as float
                trial.set_user_attr('n_examples', len(self.subject_data))

                
                self.logger.info(f"Trial {trial.number} completed with accuracy: {accuracy:.4f}")
                
                # Make sure to return the accuracy as the objective value
                return float(accuracy)
                
            finally:
                evaluator.close()
                
        except Exception as e:
            self.logger.error(f"Error in trial {trial.number}: {str(e)}", exc_info=True)
            raise
        
    def optimize(self) -> dict:
        """Run optimization process and return best parameters."""
        try:
            self.logger.info(f"\nStarting optimization for subject: {self.base_config.subject}")
            self.logger.info(f"Evaluation type: {self.eval_type}")
            
            trials = self.study.trials
            completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
            remaining_trials = self.n_trials - len(completed_trials)
            
            if remaining_trials > 0:
                self.study.optimize(self.objective, n_trials=remaining_trials)
            
            results = {
                'best_parameters': self.study.best_params,
                'best_accuracy': self.study.best_value,
                'completed_trials': len(self.study.trials),
                'model_name': self.base_config.model_name,
                'subject': self.base_config.subject,
                'study_name': self.study_name,
                'eval_type': self.eval_type
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
            
            # Include accuracy values in the trials dataframe
            accuracy_values = [t.user_attrs.get('accuracy', None) for t in self.study.trials]
            trials['accuracy'] = accuracy_values
            
            stats = {
                'number_of_trials': len(trials),
                'best_accuracy': self.study.best_value,
                'best_parameters': self.study.best_params,
                'best_trial_number': self.study.best_trial.number,
                'datetime_start': trials['datetime_start'].min(),
                'datetime_complete': trials['datetime_complete'].max(),
                'parameter_importance': optuna.importance.get_param_importances(self.study),
                'eval_type': self.eval_type,
                'accuracy_mean': trials['accuracy'].mean() if 'accuracy' in trials else None,
                'accuracy_std': trials['accuracy'].std() if 'accuracy' in trials else None
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
    parser.add_argument(
        '--eval-type',
        type=str,
        choices=['constrained', 'unconstrained'],
        default='unconstrained',
        help='Type of evaluation to run'
    )
    args = parser.parse_args()

    # Set paths based on eval_type
    eval_dir = Path("mmlu_pro_constrained" if args.eval_type == "constrained" else "mmlu_pro")
    db_name = "mmlu_optimization.db"
    storage_url = f"sqlite:///{str(eval_dir / db_name)}"

    if args.list_studies:
        studies = optuna.study.get_all_study_summaries(storage_url)
        print("\nExisting studies:")
        for study in studies:
            n_trials = study.n_trials
            eval_type = study.user_attrs.get('eval_type', 'unknown')
            print(f"- {study.study_name}: {n_trials} trials, type: {eval_type}, best value: {study.best_trial.value if study.best_trial else 'N/A'}")
        return

    try:
        # Use appropriate config based on eval_type
        base_config = CONSTRAINED_DEFAULT if args.eval_type == "constrained" else UNCONSTRAINED_DEFAULT
        
        optimizer = MMPROOptimizer(
            base_config=base_config,
            eval_type=args.eval_type,
            n_trials=20,
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

