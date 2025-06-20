import argparse
import datasets
import json
import os
import re
import pickle
from datasets import load_dataset
from huggingface_hub import login
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

from utils import gen_answer, load_model_tokenizer
from compute_th import compute


def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Evaluation of AI detection model on a validation set")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Hugging Face model path to use")

    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory to save results (default: results)")

    return parser.parse_args()



def load_benchmark() -> datasets.Dataset:
    test_set = load_dataset(f'MarcoMurgia97/Threshold_Optimization_Set')['test']

    list_of_ds = []

    test_set_human = test_set.filter(
        lambda example: (example["model"] == 'human'))
    test_set_ai = test_set.filter(
        lambda example: (example["model"] != 'human'))

    list_of_ds.append(test_set_human)
    list_of_ds.append(test_set_ai)

    tmp_dataset = datasets.concatenate_datasets(list_of_ds)
    return tmp_dataset


def load_data(file_path: str) -> Dict:
    """
    Args:
        file_path (str): Path of the JSON file to load

    Returns:
        Dict: Dictionary with loaded data, or empty dictionary in case of error
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f'Results file not found at {file_path}, creating a new one.')
        return {}
    except Exception as e:
        print(f'Error loading file {file_path}: {e}')
        return {}


def preprocess_text(text: str, attack_type: Optional[str]) -> str:
    """
    Args:
        text (str): Text to preprocess
        attack_type (str, optional): Attack type applied to the text

    Returns:
        str: Preprocessed text
    """
    if attack_type == 'whitespace':
        text = re.sub(r'[\t\n\r\f\v]', ' ', text)
    elif attack_type == 'insert_paragraphs':
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    else:
        # Default normalization for other/no attack type specified
        text = re.sub(r'\s+', ' ', text)

    return text.strip()


def create_prompt(text: str) -> str:
    """
    Create a prompt for the AI detection model.
    Args:
        text (str): Text to analyze

    Returns:
        str: Formatted prompt
    """
    return f"""Given the following text:

    "{text}"

    Analyze the text and determine if it was written by a human or generated by a large language model.
    Answer ONLY with "human" or "machine", without any additional comments."""


def calculate_metrics(
    TP: int, FP: int, TN: int, FN: int,
    human_sample_counter: int
) -> Dict[str, float]:
    """
    Calculate model evaluation metrics.
    Args:
        TP (int): True Positives (Machine predicted as Machine).
        FP (int): False Positives (Human predicted as Machine).
        TN (int): True Negatives (Human predicted as Human).
        FN (int): False Negatives (Machine predicted as Human).
        human_sample_counter (int): Total number of human examples.

    Returns:
        Dict[str, float]: Dictionary containing standard metrics.
    """
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    FPR = FP / human_sample_counter if human_sample_counter > 0 else 0.0

    return {
        'recall': recall,
        'precision': precision,
        'accuracy': accuracy,
        'f1_score': f1_score,
        'FPR': FPR
    }


def evaluate_model(
    model_path: str,
    results_dir: str
) -> None:
    """
    Evaluate the model on the validation dataset.

    Args:
        model_path (str): Path of the model to evaluate.
        results_dir (str): Directory to save results.
    """

    # Load model and tokenizer
    print(f"Loading model and tokenizer from: {model_path}")
    model, tokenizer = load_model_tokenizer(model_path)

    model_name_slug = model_path.replace('MarcoMurgia97/', '')

    results_val_dir = os.path.join(results_dir, "results_val")
    labels_scores_dir = os.path.join(results_dir, "labels_scores")

    os.makedirs(results_val_dir, exist_ok=True)
    os.makedirs(labels_scores_dir, exist_ok=True)

    results_path = os.path.join(results_val_dir, f"results_{model_name_slug}.json")
    pickle_path = os.path.join(labels_scores_dir, f"data_{model_name_slug}.pkl") # For scores/labels

    results_dict = load_data(results_path)

    # The function name is kept 'load_benchmark' for structural similarity
    benchmark_dataset = load_benchmark()

    TP, FP, TN, FN = 0, 0, 0, 0
    human_sample_counter = 0
    scores = [] # For storing probabilities
    labels = [] # For storing true labels (0=human, 1=machine)

    print(f"\n--- Evaluating model {model_path} on validation set ---")
    for example in tqdm(benchmark_dataset, desc="Processing validation set"):
        text = example['generation']
        attack = example.get('attack') # Get attack type if available
        processed_text = preprocess_text(text, attack)

        is_human_truth = example['model'] == 'human'
        if is_human_truth:
            human_sample_counter += 1

        prompt = create_prompt(processed_text)

        
        answer, prob = gen_answer(model, tokenizer, prompt, threshold=None)

        scores.append(prob)
        labels.append(0 if is_human_truth else 1)

        is_machine_prediction = 'machine' in answer.lower()
        is_machine_truth = not is_human_truth # If not human, it's machine

        if is_machine_prediction and is_machine_truth:
            TP += 1
        elif not is_machine_prediction and is_machine_truth:
            FN += 1
        elif is_machine_prediction and not is_machine_truth:
            FP += 1
        else: # not is_machine_prediction and not is_machine_truth (i.e., TN)
            TN += 1

    metrics = calculate_metrics(TP, FP, TN, FN, human_sample_counter)


    key = 'results_validation' 
    results_dict.setdefault(key, {})
    results_dict[key]['recall'] = round(metrics['recall'] * 100, 4)
    results_dict[key]['precision'] = round(metrics['precision'] * 100, 4)
    results_dict[key]['accuracy'] = round(metrics['accuracy'] * 100, 4)
    results_dict[key]['f1_score'] = round(metrics['f1_score'] * 100, 4)
    results_dict[key]['FPR'] = round(metrics['FPR'] * 100, 4)
    results_dict[key]['TP'] = TP
    results_dict[key]['FP'] = FP
    results_dict[key]['TN'] = TN
    results_dict[key]['FN'] = FN
    results_dict[key]['th'] = compute(labels, scores) 

    print("\nFinal results dictionary content:")
    print(json.dumps(results_dict, indent=2))

    print(f"Saving scores and labels to: {pickle_path}")
    with open(pickle_path, 'wb') as f:
        pickle.dump((scores, labels), f)

    print(f"Saving results JSON to: {results_path}")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)

    print("\nValidation evaluation complete.")


def main():
    """
    Main function that coordinates the script execution.
    (Identical structure to evaluation.py)
    """
    # Parse arguments
    args = parse_arguments()

    evaluate_model(
        model_path=args.model_path,
        results_dir=args.results_dir
    )


if __name__ == "__main__":
    main()
