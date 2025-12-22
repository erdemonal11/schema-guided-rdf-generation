import sys
import os
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

REAL_DATA_PATH = Path("data/processed/kg_triples_ids.txt")
SYNTHETIC_DIR = Path("data/synthetic")

def evaluate():
    if not list(SYNTHETIC_DIR.glob("generated_*.txt")):
        print("No synthetic data found.")
        return

    latest_file = sorted(SYNTHETIC_DIR.glob("generated_*.txt"))[-1]
    
    real_triples = set()
    with open(REAL_DATA_PATH, 'r', encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                real_triples.add(f"{parts[0]}\t{parts[1]}\t{parts[2]}")

    synthetic_triples = []
    with open(latest_file, 'r', encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                synthetic_triples.append(f"{parts[0]}\t{parts[1]}\t{parts[2]}")

    total_gen = len(synthetic_triples)
    if total_gen == 0:
        return

    novel_count = sum(1 for triple in synthetic_triples if triple not in real_triples)
    overlap_count = total_gen - novel_count
            
    novelty_score = (novel_count / total_gen) * 100
    overlap_score = (overlap_count / total_gen) * 100
    unique_gen = len(set(synthetic_triples))
    uniqueness_score = (unique_gen / total_gen) * 100

    print(f"File: {latest_file.name}")
    print(f"Total: {total_gen}")
    print(f"Novelty: {novelty_score:.2f}%")
    print(f"Train Overlap: {overlap_score:.2f}%")
    print(f"Uniqueness: {uniqueness_score:.2f}%")

if __name__ == "__main__":
    evaluate()