from evaluation.mmlu_pro.config import ModelConfig, DEFAULT_CONFIG
from evaluation.mmlu_pro.eval import main
import optuna
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a copy of the default config
config = DEFAULT_CONFIG

def objective(trial):
    # Sample parameters from the search space
    config.temperature = trial.suggest_float('temperature', 0.1, 0.9)
    config.top_p = trial.suggest_float('top_p', 0.1, 0.9)
    
    # Run evaluation with these parameters
    try:
        accuracy = main()
        logger.info(f"Trial {trial.number} - Temperature: {config.temperature:.3f}, "
                   f"Top_p: {config.top_p:.3f}, Accuracy: {accuracy:.4f}")
        return accuracy
    except Exception as e:
        logger.error(f"Trial failed with error: {str(e)}")
        return float('-inf')

def optimize_parameters(n_trials=20):
    # Create a study object with storage
    study = optuna.create_study(
        study_name="mmlu_parameter_optimization",
        storage="sqlite:///optuna_results.db",  # Save results to SQLite database
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True  # Allow resuming existing study
    )
    
    # Run optimization
    logger.info(f"Starting optimization with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials)
    
    # Get best parameters and results
    best_params = study.best_params
    best_accuracy = study.best_value
    
    # Print results
    print("\nOptimization Results:")
    print("=" * 50)
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print(f"Best Parameters:")
    print(f"- Temperature: {best_params['temperature']:.3f}")
    print(f"- Top_p: {best_params['top_p']:.3f}")
    
    return best_params, best_accuracy

if __name__ == "__main__":
    # Run optimization
    best_params, best_accuracy = optimize_parameters(n_trials=20)