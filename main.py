#!/usr/bin/env python3
"""
Parrot GPT - Unified CLI
------------------------
Entry point for all Parrot GPT operations.
Dispatches commands to scripts in src/.
"""

import sys
import argparse
from pathlib import Path
import importlib

# Add src to python path to allow imports to work
SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))

def run_module(module_name):
    """Import and run the main function of a module."""
    try:
        # Import the module dynamically
        module = importlib.import_module(module_name)
        
        # Check if it has a main function
        if hasattr(module, 'main'):
            module.main()
        else:
            print(f"Error: Module '{module_name}' has no main() function.")
            sys.exit(1)
            
    except ImportError as e:
        print(f"Error importing module '{module_name}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error running '{module_name}': {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Parrot GPT - Minimal GPT Implementation",
        usage="""python main.py <command> [<args>]

Available commands:
   train       Train the model
   generate    Generate text from a trained model
   tokenize    Train tokenizer and encode data
   preprocess  Clean and split raw text
   test        Run architecture tests
   
Use 'python main.py <command> --help' for help on a specific command.
"""
    )
    
    parser.add_argument('command', help='Subcommand to run')
    
    # Parse the first argument (command) and ignore the rest for now
    args = parser.parse_args(sys.argv[1:2])
    
    # Dispatch to appropriate module
    # We remove the first argument (main.py) and the command name
    # leaving only the arguments for the subcommand
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    
    commands = {
        'train': 'train',
        'generate': 'generate',
        'tokenize': 'tokenize',
        'preprocess': 'preprocess_data',
        'test': 'test_model',
        'create': 'create_model'
    }
    
    if args.command in commands:
        run_module(commands[args.command])
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
