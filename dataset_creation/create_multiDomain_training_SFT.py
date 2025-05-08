#!/usr/bin/env python3
"""
Dataset Balancer for AI vs Human Text Classification

This script prepares balanced datasets from the RAID corpus for training text classification models
to distinguish between human and AI-generated content across various domains.
"""

import json
import argparse
import random
import re
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from typing import List, Tuple, Dict

def parse_arguments() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(description="Balance and prepare datasets for AI detection fine-tuning")
    
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the processed dataset")
    parser.add_argument("--human_samples", type=int, default=1520,
                        help="Number of human samples per domain")
    parser.add_argument("--ai_samples_per_model", type=int, default=252,
                        help="Number of samples per AI model per domain")
    parser.add_argument("--model_format", type=str, choices=["LLama3"], default="LLama3",
                        help="Target model format for prompts")
    parser.add_argument("--domains", nargs="+", 
                        default=["abstracts", "wiki", "reddit", "recipes", "news", "poetry", "books"],
                        help="Domains to include in the dataset")
    
    return parser.parse_args()

def balance_dataset_human(raid_dataset, domains: List[str], samples_per_domain: int):
    """
    Create a balanced dataset from human-written content across specified domains.
    
    Args:
        raid_dataset: The source RAID dataset
        domains: List of domains to include
        samples_per_domain: Number of samples to include per domain
    
    Returns:
        A balanced dataset with human content
    """
    list_of_datasets = []
    human_dataset = raid_dataset.filter(lambda example: example['model'] == 'human' and example['attack'] == 'none')
    
    print("Balancing human dataset across domains:")
    for domain in domains:
        domain_dataset = human_dataset.filter(lambda example: example['domain'] == domain.lower())
        
        if len(domain_dataset) < samples_per_domain:
            print(f"Warning: Only {len(domain_dataset)} samples available for human domain '{domain}' (requested {samples_per_domain})")
            list_of_datasets.append(domain_dataset)
        else:
            list_of_datasets.append(domain_dataset.select(range(samples_per_domain)))
            
        # Print some example titles to verify content
        print(f"Domain: {domain} - Example titles:")
        for i in range(0, samples_per_domain, samples_per_domain//5)[:5]:
            if i < len(domain_dataset):
                print(f"  - {domain_dataset[i]['title']}")
        print("------")
    
    return concatenate_datasets(list_of_datasets)

def balance_dataset_ai(raid_dataset, domains: List[str], samples_per_model: int):
    """
    Create a balanced dataset from AI-generated content across specified domains and models.
    
    Args:
        raid_dataset: The source RAID dataset
        domains: List of domains to include
        samples_per_model: Number of samples to include per AI model per domain
    
    Returns:
        A balanced dataset with AI-generated content
    """
    list_of_datasets = []
    ai_dataset = raid_dataset.filter(lambda example: example["model"] != 'human' and example['attack'] == 'none')
    
    # Define models to extract with their filtering conditions
    model_configs = [
        {
            "name": "mistral-chat",
            "filter": lambda ex, domain: (
                ex['model'] == 'mistral-chat' and ex['domain'] == domain.lower() and
                ex['decoding'] == 'sampling' and ex['repetition_penalty'] == 'yes'
            ),
            "range_start": 0,
            "range_end": samples_per_model
        },
        {
            "name": "cohere-chat",
            "filter": lambda ex, domain: (
                ex['model'] == 'cohere-chat' and ex['domain'] == domain.lower() and
                ex['decoding'] == 'sampling'
            ),
            "range_start": samples_per_model,
            "range_end": samples_per_model * 2
        },
        {
            "name": "gpt4",
            "filter": lambda ex, domain: (
                ex['model'] == 'gpt4' and ex['domain'] == domain.lower() and
                ex['decoding'] == 'sampling'
            ),
            "range_start": samples_per_model * 2,
            "range_end": samples_per_model * 3
        },
        {
            "name": "mistral",
            "filter": lambda ex, domain: (
                ex['model'] == 'mistral' and ex['domain'] == domain.lower() and
                ex['decoding'] == 'sampling' and ex['repetition_penalty'] == 'yes'
            ),
            "range_start": samples_per_model * 3,
            "range_end": samples_per_model * 4
        },
        {
            "name": "cohere",
            "filter": lambda ex, domain: (
                ex['model'] == 'cohere' and ex['domain'] == domain.lower() and
                ex['decoding'] == 'sampling'
            ),
            "range_start": samples_per_model * 4,
            "range_end": samples_per_model * 5
        },
        {
            "name": "gpt3",
            "filter": lambda ex, domain: (
                ex['model'] == 'gpt3' and ex['domain'] == domain.lower() and
                ex['decoding'] == 'sampling'
            ),
            "range_start": samples_per_model * 5,
            "range_end": samples_per_model * 6
        }
    ]
    
    print("Balancing AI dataset across domains and models:")
    for domain in domains:
        print(f"Processing domain: {domain}")
        
        for model_config in model_configs:
            filtered_dataset = ai_dataset.filter(
                lambda example: model_config["filter"](example, domain)
            )
            
            if len(filtered_dataset) < samples_per_model:
                print(f"Warning: Only {len(filtered_dataset)} samples available for model '{model_config['name']}' in domain '{domain}' (requested {samples_per_model})")
                if len(filtered_dataset) > 0:
                    list_of_datasets.append(filtered_dataset)
            else:
                range_start = model_config["range_start"]
                range_end = model_config["range_end"]
                list_of_datasets.append(filtered_dataset.select(range(min(samples_per_model, len(filtered_dataset)))))
    
    return concatenate_datasets(list_of_datasets)

def get_balanced_datasets(raid_dataset, domains: List[str], human_samples: int, ai_samples_per_model: int) -> Tuple:
    """
    Create balanced datasets for both human and AI-generated content.
    
    Args:
        raid_dataset: The source RAID dataset
        domains: List of domains to include
        human_samples: Number of human samples per domain
        ai_samples_per_model: Number of samples per AI model per domain
    
    Returns:
        Tuple containing (human_dataset, ai_dataset)
    """
    raid_filtered_human = balance_dataset_human(raid_dataset, domains, human_samples)
    raid_filtered_ai = balance_dataset_ai(raid_dataset, domains, ai_samples_per_model)
    
    return raid_filtered_human, raid_filtered_ai

def prepare_prompts(dataset):
    """
    Prepare training prompts in the specified model format.
    
    Args:
        dataset: Dataset to prepare prompts from
    
    Returns:
        List of formatted prompts
    """
    formatted_prompts = []
    
    for example in dataset:
        # Determine the correct answer
        answer = 'human' if example['model'] == 'human' else 'machine'
        
        # Clean the text
        text_to_detect = example['generation']
        text_to_detect = re.sub(r'\s+', ' ', text_to_detect)
        text_to_detect = text_to_detect.strip()
        
        # Create the instruction
        instruction = f"""Given the following text:

        "{text_to_detect}"

        Analyze the text and determine if it was written by a human or generated by a large language model.
        Answer ONLY with "human" or "machine", without any additional comments."""
        
        # Format according to model requirements
        prompt = f"<|start_header_id|>user<|end_header_id|>{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{answer}<|eot_id|>"
        
        formatted_prompts.append({'text': prompt})
    
    return formatted_prompts

def format_training_set(human_dataset, ai_dataset):
    """
    Combine human and AI datasets into a single training set with proper formatting.
    
    Args:
        human_dataset: Dataset with human-written content
        ai_dataset: Dataset with AI-generated content
    
    Returns:
        Combined and formatted training set
    """
    # Prepare prompts for both datasets
    human_prompts = prepare_prompts(human_dataset)
    ai_prompts = prepare_prompts(ai_dataset)
    
    # Combine and shuffle
    training_set = human_prompts + ai_prompts
    random.shuffle(training_set)
    
    return training_set

def main():
    """Main function to execute the dataset balancing and preparation workflow."""
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the dataset
    print(f"Loading RAID dataset")
    raid_train = load_dataset('MarcoMurgia97/raid_training_set')['train']
    
    # Get balanced datasets
    print(f"Creating balanced datasets with {args.human_samples} human samples per domain and {args.ai_samples_per_model} samples per AI model per domain")
    dataset_human, dataset_ai = get_balanced_datasets(
        raid_train, 
        args.domains, 
        args.human_samples, 
        args.ai_samples_per_model
    )
    
    # Save the combined dataset to disk
    combined_dataset = concatenate_datasets([dataset_human, dataset_ai])
    dataset_path = output_dir / "balanced_dataset"
    combined_dataset.save_to_disk(dataset_path)
    print(f"Saved combined dataset to {dataset_path}")
    
    # Format the training set
    training_set = format_training_set(dataset_human, dataset_ai)
    
    # Define a descriptive name for the JSON file
    domains_str = "_".join(args.domains)
    total_samples = len(training_set) // 1000
    
    json_filename = f"dataset_for_tuning_{total_samples}k_samples_{domains_str}.json"
    json_path = output_dir / json_filename
    
    # Save the formatted training set as JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(training_set, f, indent=2, ensure_ascii=False)
    
    print(f"Saved formatted training set to {json_path}")
    print(f"Total samples: {len(training_set)}")

if __name__ == '__main__':
    main()