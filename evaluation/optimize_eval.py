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
from typing import Optional, Union, Set, Tuple
import argparse
from urllib.parse import quote
import time
import gc
import signal
import psutil
import os

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

def get_last_completed_trial(study: optuna.Study) -> Optional[optuna.Trial]:
    """Get last completed trial"""
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    return max(completed_trials, key=lambda t: t.number) if completed_trials else None

def cleanup_memory():
    """Clean up GPU and CPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    
def check_temperature() -> bool:
    """Check GPU and CPU temperatures"""
    try:
        # Check GPU temperature using nvidia-smi
        gpu_temp = float(os.popen('nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits').read())
        if gpu_temp > 84:  # Threshold in Celsius
            return False
            
        # Check CPU temperature if available
        temps = psutil.sensors_temperatures()
        if 'coretemp' in temps:
            cpu_temp = max(temp.current for temp in temps['coretemp'])
            if cpu_temp > 85:  # Threshold in Celsius
                return False
                
        return True
    except:
        return True  # Continue if temperature check fails

def continue_optimization(study: optuna.Study, n_trials: int) -> optuna.Study:
    """Continue optimization with temperature checks and delays"""
    last_trial = get_last_completed_trial(study)
    if not last_trial:
        return study
        
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.RUNNING:
            trial._state = optuna.trial.TrialState.FAIL
    
    return study

def get_tried_params(study: optuna.Study) -> Set[Tuple[float, float]]:
    """Get set of previously tried (temperature, top_p) combinations"""
    return {
        (t.params['temperature'], t.params['top_p']) 
        for t in study.trials 
        if t.state == optuna.trial.TrialState.COMPLETE
    }

class MMPROOptimizer:
    def __init__(
        self,
        base_config: Union[UnconstrainedConfig, ConstrainedConfig],
        eval_type: str = "unconstrained",
        n_trials: int = 20,
        storage_url: Optional[str] = None,
        study_name: Optional[str] = None,
        study: Optional[optuna.Study] = None
    ):
        self.base_config = base_config
        self.eval_type = eval_type
        self.n_trials = n_trials
        self.storage_url = storage_url
        self.study_name = study_name or f"mmlu_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger = logging.getLogger(__name__)

        if study:
            self.study = study
            self.logger.info(f"Using existing study with {len(study.trials)} trials")
        else:
            self.study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage_url,
                direction="maximize",
                load_if_exists=False  # Don't load if exists, we want new study
            )

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
            
            # Configure TPESampler with specific settings
            sampler = optuna.samplers.TPESampler(
                seed=42,  # Fixed seed for reproducibility
                n_startup_trials=5,  # Initial random trials for exploration
                multivariate=True,  # Consider parameter relationships
                constant_liar=True  # Handle parallel optimization better
            )
            
            self.study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage_url,
                direction="maximize",
                sampler=sampler,
                load_if_exists=True
            )
            
            # Seed study with known good parameter combinations if new study
            if not study_exists:
                initial_trials = [
                    {'temperature': 0.5, 'top_p': 0.5},  # Balanced
                    {'temperature': 0.3, 'top_p': 0.7},  # Conservative temp, high diversity
                    {'temperature': 0.7, 'top_p': 0.3}   # High temp, low diversity
                ]
                for trial_params in initial_trials:
                    # Use suggest_float without step parameter
                    def enqueue_trial(trial):
                        trial.suggest_float('temperature', 0.01, 0.95)
                        trial.suggest_float('top_p', 0.01, 0.95)
                        return 0.0
                    self.study.enqueue_trial(trial_params)

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
        """Setup evaluation environment"""
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
            self.tokenizer.padding_side = 'left'
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

    def _create_evaluator(self, config: Union[UnconstrainedConfig, ConstrainedConfig]):
        """Create appropriate evaluator based on eval_type"""
        try:
            if self.eval_type == 'constrained':
                if not hasattr(self, 'tokenizer'):
                    raise ValueError("Tokenizer not initialized")
                return MMLPROEvaluatorConstr(
                    tokenizer=self.tokenizer,
                    config=config,
                    model_name=self.base_config.model_name
                )
            elif self.eval_type == 'unconstrained':
                return MMPROEvaluator(config)
            else:
                raise ValueError(f"Unknown eval_type: {self.eval_type}")
        except Exception as e:
            self.logger.error(f"Error creating evaluator: {str(e)}")
            raise

    def objective(self, trial: optuna.Trial) -> float:
        """Run single optimization trial with continuous parameter sampling"""
        try:
            max_attempts = 50
            for attempt in range(max_attempts):
                # Remove step parameter for continuous sampling
                temp = trial.suggest_float('temperature', 0.01, 0.95)
                top_p = trial.suggest_float('top_p', 0.01, 0.95)
                
                # Round for display/comparison only
                temp_rounded = round(temp, 1)
                top_p_rounded = round(top_p, 1)
                
                tried_params = get_tried_params(self.study)
                if (temp_rounded, top_p_rounded) not in tried_params:
                    break
                    
                trial._params.clear()
                
                if attempt == max_attempts - 1:
                    raise RuntimeError("Could not find unused parameter combination")

            trial_config = deepcopy(self.base_config)
            trial_config.temperature = temp  # Use unrounded values
            trial_config.top_p = top_p

            self.logger.info(f"\nStarting trial {trial.number}")
            self.logger.info(f"Trial {trial.number} parameters:")
            self.logger.info(f"Temperature: {trial_config.temperature:.1f}")
            self.logger.info(f"Top_p: {trial_config.top_p:.1f}")

            # Store trial information
            trial.set_user_attr('model_name', self.base_config.model_name)
            trial.set_user_attr('subject', self.base_config.subject)
            trial.set_user_attr('timestamp', datetime.now().isoformat())
            trial.set_user_attr('eval_type', self.eval_type)

            # Run evaluation
            evaluator = self._create_evaluator(trial_config)
            try:
                results = evaluator.evaluate(self.subject_data)
                accuracy = results['accuracy']
                trial.set_user_attr('accuracy', float(accuracy))
                self.logger.info(f"Trial {trial.number} completed with accuracy: {accuracy:.4f}")
                return float(accuracy)
            finally:
                evaluator.close()

        except Exception as e:
            self.logger.error(f"Error in trial {trial.number}: {str(e)}", exc_info=True)
            raise
        
    def optimize(self) -> dict:
        """Run optimization process and return best parameters."""
        try:
            # Continue from last completed trial
            self.study = continue_optimization(self.study, self.n_trials)
            last_trial = get_last_completed_trial(self.study)
            
            if last_trial:
                self.logger.info(f"Resuming from trial {last_trial.number + 1}")
                self.logger.info(f"Best accuracy so far: {self.study.best_value:.4f}")
            
            # Optimize remaining trials
            remaining_trials = self.n_trials - len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            
            for _ in range(remaining_trials):
                # Check temperature before each trial
                if not check_temperature():
                    self.logger.warning("System temperature too high, pausing for 60s")
                    time.sleep(10)
                    continue
                    
                self.study.optimize(
                    self.objective,
                    n_trials=1,  # Run one trial at a time
                    timeout=None
                )
                
                # Cleanup and cooling period
                cleanup_memory()
                time.sleep(3)  # Cool-down period between trials
            
            return {
                'best_parameters': self.study.best_params,
                'best_accuracy': self.study.best_value,
                'completed_trials': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                'model_name': self.base_config.model_name,
                'subject': self.base_config.subject,
                'study_name': self.study_name,
                'eval_type': self.eval_type
            }
            
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

def verify_study_exists(storage_url: str, study_name: str, eval_type: str) -> bool:
    """Verify if study exists and matches eval_type"""
    studies = optuna.study.get_all_study_summaries(storage=storage_url)
    study_match = next((s for s in studies if s.study_name == study_name), None)
    
    if not study_match:
        return False
        
    # For existing studies without eval_type, allow continuation
    study_eval_type = study_match.user_attrs.get('eval_type')
    return study_eval_type is None or study_eval_type == eval_type

def display_study_details(storage_url: str, pattern: str = None, verbose: bool = False):
    """Display study information with optional pattern matching and verbosity"""
    studies = optuna.study.get_all_study_summaries(storage=storage_url)
    
    # Filter studies if pattern provided
    if pattern:
        studies = [s for s in studies if pattern.lower() in s.study_name.lower()]
    
    if not studies:
        print(f"No studies found{' matching pattern: ' + pattern if pattern else ''}")
        return

    if not verbose:
        print("\nExisting studies:")
        for study in studies:
            print(f"- {study.study_name}: {study.n_trials} trials completed")
        print("\nUse --verbose flag for detailed trial information")
        return

    for study in studies:
        print(f"\nStudy: {study.study_name}")
        print("=" * (len(study.study_name) + 7))
        print(f"Subject: {study.user_attrs.get('subject', 'Not set')}")
        print(f"Eval Type: {study.user_attrs.get('eval_type', 'Not set')}")
        
        full_study = optuna.load_study(study_name=study.study_name, storage=storage_url)
        
        print("\nTrial Results:")
        print("-" * 80)
        print(f"{'Trial':^6} | {'Temperature':^11} | {'Top_p':^7} | {'Accuracy':^10} | {'State':^12}")
        print("-" * 80)
        
        for trial in full_study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                acc = trial.user_attrs.get('accuracy', 'N/A')
                acc_str = f"{acc:.4f}" if isinstance(acc, float) else acc
                print(f"{trial.number:^6} | {trial.params.get('temperature', 'N/A'):^11.1f} | "
                      f"{trial.params.get('top_p', 'N/A'):^7.1f} | {acc_str:^10} | {trial.state.name:^12}")
            else:
                print(f"{trial.number:^6} | {'N/A':^11} | {'N/A':^7} | {'N/A':^10} | {trial.state.name:^12}")
        print("-" * 80)
        print(f"Best accuracy: {study.best_trial.value:.4f}" if study.best_trial else "No completed trials")
        print()

def main():
    parser = argparse.ArgumentParser(description='MMLU optimization utilities')
    parser.add_argument('--continue-study', type=str, help='Name of study to continue') 
    parser.add_argument('--list-studies', type=str, nargs='?', const='', help='List studies, optionally filter by pattern')
    parser.add_argument('--verbose', action='store_true', help='Show detailed trial information')
    parser.add_argument(
        '--eval-type',
        type=str,
        choices=['constrained', 'unconstrained'],
        default='unconstrained',
        help='Type of evaluation to run'
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default=None,
        help='Path to store/load optimization database'
    )
    parser.add_argument(
        '--search-study',
        type=str,
        help='Search for studies matching pattern'
    )
    args = parser.parse_args()

    # Set paths based on eval_type and db_path
    if args.db_path:
        db_path = Path(args.db_path)
    else:
        eval_dir = Path("mmlu_pro_constrained" if args.eval_type == "constrained" else "mmlu_pro")
        db_path = eval_dir / "mmlu_optimization.db"
    
    # Create parent directories if they don't exist
    db_path.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{str(db_path)}"

    if args.list_studies is not None:
        display_study_details(storage_url, args.list_studies, args.verbose)
        return

    if args.search_study:
        display_study_details(storage_url, args.search_study)
        return

    try:
        if args.continue_study:
            # First verify study exists
            studies = optuna.study.get_all_study_summaries(storage=storage_url)
            study_match = next((s for s in studies if s.study_name == args.continue_study), None)
            
            if not study_match:
                logger.error(f"Study '{args.continue_study}' not found. Available studies:")
                for s in studies:
                    logger.info(f"- {s.study_name}")
                raise ValueError("Study not found")
            
            # Load existing study
            study = optuna.load_study(
                study_name=args.continue_study,
                storage=storage_url
            )
            logger.info(f"Loaded existing study: {args.continue_study}")
            logger.info(f"Number of trials: {len(study.trials)}")
            
            # Create optimizer with existing study
            optimizer = MMPROOptimizer(
                base_config=CONSTRAINED_DEFAULT if args.eval_type == "constrained" else UNCONSTRAINED_DEFAULT,
                eval_type=args.eval_type,
                n_trials=20,
                study=study,
                study_name=args.continue_study,  # Pass existing study name
                storage_url=storage_url
            )
        else:
            # Create new study with timestamp
            study_name = f"mmlu_opt_{args.eval_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            optimizer = MMPROOptimizer(
                base_config=CONSTRAINED_DEFAULT if args.eval_type == "constrained" else UNCONSTRAINED_DEFAULT,
                eval_type=args.eval_type,
                n_trials=20,
                study_name=study_name,
                storage_url=storage_url
            )

        results = optimizer.optimize()
        
        print(f"\nBest parameters found:")
        print(f"Temperature: {results['best_parameters']['temperature']:.1f}")
        print(f"Top_p: {results['best_parameters']['top_p']:.1f}")
        print(f"Best accuracy: {results['best_accuracy']:.4f}")
        
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

