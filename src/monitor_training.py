#!/usr/bin/env python3
"""
Monitor training progress in real-time.
Reads the training log and displays progress.
"""

import argparse
import time
from pathlib import Path
import pandas as pd
import numpy as np


def print_progress_bar(current, total, length=40):
    """Print a progress bar."""
    filled = int(length * current / total)
    bar = '█' * filled + '░' * (length - filled)
    percent = 100 * current / total
    return f"[{bar}] {percent:.1f}%"


def monitor_training(log_file: Path, refresh_interval: int = 5):
    """
    Monitor training progress in real-time.
    
    Args:
        log_file: Path to training_log.csv
        refresh_interval: Seconds between updates
    """
    print("="*70)
    print("TRAINING MONITOR")
    print("="*70)
    print(f"Monitoring: {log_file}")
    print(f"Press Ctrl+C to stop")
    print("="*70)
    print()
    
    last_step = 0
    max_steps = None
    
    try:
        while True:
            if not log_file.exists():
                print("Waiting for training to start...")
                time.sleep(refresh_interval)
                continue
            
            # Read log file
            try:
                df = pd.read_csv(log_file)
                
                if len(df) == 0:
                    print("Waiting for first log entry...")
                    time.sleep(refresh_interval)
                    continue
                
                # Get latest stats
                latest = df.iloc[-1]
                step = int(latest['step'])
                train_loss = latest['train_loss']
                val_loss = latest['val_loss'] if 'val_loss' in df.columns and not pd.isna(latest['val_loss']) else None
                lr = latest['lr']
                elapsed_time = latest['time']
                
                # Estimate max steps if not known
                if max_steps is None:
                    # Try to infer from log interval
                    if len(df) > 1:
                        step_interval = df['step'].diff().mode()[0]
                        # Assume eval_interval is 500 (typical)
                        max_steps = 15000  # Default assumption
                
                # Calculate derived stats
                perplexity = np.exp(train_loss)
                
                if step > last_step:
                    # Clear and redraw
                    print("\033[2J\033[H")  # Clear screen
                    
                    print("="*70)
                    print("TRAINING PROGRESS")
                    print("="*70)
                    print()
                    
                    # Progress bar
                    if max_steps:
                        progress = print_progress_bar(step, max_steps)
                        print(f"Progress: {progress}  ({step:,}/{max_steps:,} steps)")
                    else:
                        print(f"Steps: {step:,}")
                    print()
                    
                    # Current metrics
                    print(f"{'Metric':<20} {'Value':>15}")
                    print("-"*40)
                    print(f"{'Training Loss':<20} {train_loss:>15.4f}")
                    if val_loss is not None:
                        print(f"{'Validation Loss':<20} {val_loss:>15.4f}")
                    print(f"{'Perplexity':<20} {perplexity:>15.2f}")
                    print(f"{'Learning Rate':<20} {lr:>15.2e}")
                    print()
                    
                    # Time stats
                    hours = elapsed_time / 3600
                    if max_steps and step > 0:
                        time_per_step = elapsed_time / step
                        remaining_steps = max_steps - step
                        eta_seconds = time_per_step * remaining_steps
                        eta_hours = eta_seconds / 3600
                        
                        print(f"{'Time Stats':<20} {'Value':>15}")
                        print("-"*40)
                        print(f"{'Elapsed':<20} {hours:>14.2f}h")
                        print(f"{'ETA':<20} {eta_hours:>14.2f}h")
                        print(f"{'Steps/hour':<20} {3600/time_per_step:>15,.0f}")
                    else:
                        print(f"Elapsed time: {hours:.2f} hours")
                    print()
                    
                    # Training history (last 10 evaluations)
                    if len(df) > 1:
                        print("Recent History:")
                        print("-"*70)
                        recent = df.tail(10)
                        for _, row in recent.iterrows():
                            s = int(row['step'])
                            tl = row['train_loss']
                            vl = row['val_loss'] if not pd.isna(row['val_loss']) else None
                            vl_str = f"{vl:.4f}" if vl is not None else "N/A    "
                            print(f"  Step {s:5d} | Train: {tl:.4f} | Val: {vl_str}")
                    print()
                    
                    # Status
                    print("="*70)
                    print(f"Last update: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"Monitoring... (refreshing every {refresh_interval}s)")
                    print("="*70)
                    
                    last_step = step
                
                time.sleep(refresh_interval)
                
            except pd.errors.EmptyDataError:
                print("Waiting for training data...")
                time.sleep(refresh_interval)
            except Exception as e:
                print(f"Error reading log: {e}")
                time.sleep(refresh_interval)
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        if log_file.exists():
            df = pd.read_csv(log_file)
            if len(df) > 0:
                print("\nFinal Stats:")
                latest = df.iloc[-1]
                print(f"  Steps completed: {int(latest['step']):,}")
                print(f"  Final train loss: {latest['train_loss']:.4f}")
                if 'val_loss' in df.columns:
                    val_loss = df['val_loss'].dropna().iloc[-1]
                    print(f"  Final val loss: {val_loss:.4f}")
                    print(f"  Final perplexity: {np.exp(val_loss):.2f}")
                print(f"  Total time: {latest['time']/3600:.2f} hours")


def main():
    parser = argparse.ArgumentParser(description='Monitor training progress')
    parser.add_argument(
        '--log-file',
        type=str,
        default='logs/training_log.csv',
        help='Path to training log CSV file'
    )
    parser.add_argument(
        '--refresh',
        type=int,
        default=5,
        help='Refresh interval in seconds'
    )
    
    args = parser.parse_args()
    
    log_file = Path(args.log_file)
    monitor_training(log_file, args.refresh)


if __name__ == '__main__':
    main()