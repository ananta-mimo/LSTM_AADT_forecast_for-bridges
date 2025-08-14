# Convenience runner that mirrors the modular pipeline.
from code.train import main as train_main

if __name__ == "__main__":
    # Example: python code/main.py --csv input_data/raw/sample_adt.csv --save_model
    train_main()
