
import json
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve
from dna_detectllm import DetectLLM

def load_json_texts(human_file, ai_file):
    with open(human_file, 'r') as f:
        human_json = json.load(f)
    with open(ai_file, 'r') as f:
        ai_json = json.load(f)
    return human_json["human_text"], ai_json["machine_text"]

def compute_scores(bino, texts, label):
    scores = []
    for text in tqdm(texts, desc=f"Scoring {label} texts"):
        try:
            score = bino.compute_score(text)
        except Exception as e:
            print(f"Error scoring text: {e}")
            score = 0.0
        scores.append(score)
    return scores

def evaluate(human_scores, ai_scores):
    scores = human_scores + ai_scores
    labels = [1] * len(human_scores) + [0] * len(ai_scores)

    auroc = roc_auc_score(labels, scores)
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_f1 = np.max(f1_scores)

    return auroc, best_f1

def main():
    parser = argparse.ArgumentParser(description="Evaluate DNA-DetectLLM AUROC and F1.")
    parser.add_argument('--human_file', type=str, required=True, help="Path to human_text JSON file.")
    parser.add_argument('--ai_file', type=str, required=True, help="Path to AI_text JSON file.")
    args = parser.parse_args()

    human_texts, ai_texts = load_json_texts(args.human_file, args.ai_file)

    bino = DetectLLM()

    human_scores = compute_scores(bino, human_texts, "human")
    ai_scores = compute_scores(bino, ai_texts, "AI")

    auroc, best_f1 = evaluate(human_scores, ai_scores)

    print(f"AUROC: {auroc:.4f}")
    print(f"Best F1: {best_f1:.4f}")

if __name__ == '__main__':
    main()
