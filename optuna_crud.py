import optuna
from pathlib import Path
import argparse
from datetime import datetime
import os

def get_storage_url(db_path: str) -> str:
    """Construct storage URL from database path."""
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Database not found at: {path}")
    return f"sqlite:///{path}"

def list_studies(storage_url: str):
    """List all studies in database."""
    studies = optuna.study.get_all_study_summaries(storage_url)
    print(f"\nStudies in database: {storage_url}")
    for study in studies:
        print(f"\nStudy: {study.study_name}")
        print(f"Number of trials: {study.n_trials}")
        print(f"Best value: {study.best_trial.value if study.best_trial else 'N/A'}")

def show_study_details(storage_url: str, study_name: str):
    """Show detailed information about a specific study."""
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    
    print(f"\nStudy: {study_name}")
    print(f"Number of trials: {len(study.trials)}")
    print("\nBest trial:")
    if study.best_trial:
        print(f"  Value: {study.best_trial.value}")
        print(f"  Parameters: {study.best_trial.params}")
    
    print("\nAll trials:")
    for trial in study.trials:
        print(f"\nTrial {trial.number}:")
        print(f"  Value: {trial.value}")
        print(f"  Parameters: {trial.params}")
        print(f"  State: {trial.state}")

def delete_failed_trials(storage_url: str, study_name: str):
    """Delete failed trials from a study."""
    import sqlite3
    
    # Direct database connection for deletion
    db_path = storage_url.replace('sqlite:///', '')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # First, get the study_id
    cursor.execute("SELECT study_id FROM studies WHERE study_name = ?", (study_name,))
    study_id = cursor.fetchone()[0]
    
    # Get failed trials
    cursor.execute("""
        SELECT trial_id FROM trials 
        WHERE study_id = ? AND state = 'FAIL'
    """, (study_id,))
    failed_trial_ids = [row[0] for row in cursor.fetchall()]
    
    print(f"\nFound {len(failed_trial_ids)} failed trials")
    
    if failed_trial_ids:
        # Delete trial values
        cursor.execute("""
            DELETE FROM trial_values 
            WHERE trial_id IN ({})
        """.format(','.join('?' * len(failed_trial_ids))), failed_trial_ids)
        
        # Delete trial params
        cursor.execute("""
            DELETE FROM trial_params 
            WHERE trial_id IN ({})
        """.format(','.join('?' * len(failed_trial_ids))), failed_trial_ids)
        
        # Delete trial user attributes
        cursor.execute("""
            DELETE FROM trial_user_attributes 
            WHERE trial_id IN ({})
        """.format(','.join('?' * len(failed_trial_ids))), failed_trial_ids)
        
        # Delete trials
        cursor.execute("""
            DELETE FROM trials 
            WHERE trial_id IN ({})
        """.format(','.join('?' * len(failed_trial_ids))), failed_trial_ids)
        
        conn.commit()
        print(f"Deleted {len(failed_trial_ids)} failed trials")
    
    conn.close()

def delete_study(storage_url: str, study_name: str):
    """Delete an entire study."""
    optuna.delete_study(study_name=study_name, storage=storage_url)
    print(f"\nStudy {study_name} deleted")

def main():
    parser = argparse.ArgumentParser(description='Optuna Study Management')
    parser.add_argument('--db', type=str, required=True,
                       help='Path to the SQLite database file')
    parser.add_argument('command', choices=['list', 'show', 'clean', 'delete'],
                       help='Command to execute')
    parser.add_argument('study_name', nargs='?',
                       help='Name of the study (required for show, clean, and delete)')
    
    args = parser.parse_args()
    
    try:
        # Get storage URL from database path
        storage_url = get_storage_url(args.db)
        
        if args.command == 'list':
            list_studies(storage_url)
        elif args.command in ['show', 'clean', 'delete']:
            if not args.study_name:
                raise ValueError(f"Study name is required for {args.command} command")
            
            if args.command == 'show':
                show_study_details(storage_url, args.study_name)
            elif args.command == 'clean':
                delete_failed_trials(storage_url, args.study_name)
            elif args.command == 'delete':
                delete_study(storage_url, args.study_name)
        else:
            parser.print_help()
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()