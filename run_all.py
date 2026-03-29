import os

print("Step 1: Collecting data...")
os.system("python3 src/data_collector.py")

print("Step 2: Preprocessing data...")
os.system("python3 src/preprocess.py")

print("Step 3: Training ML model...")
os.system("pytho3 src/train_ml_model.py")

print("Step 4: Training LLM model...")
os.system("python3 src/train_llm_model.py")

print("Step 5: Generating visualization...")
os.system("python3 src/visualize.py")

print("✅ All steps completed. Check results/plots.png")