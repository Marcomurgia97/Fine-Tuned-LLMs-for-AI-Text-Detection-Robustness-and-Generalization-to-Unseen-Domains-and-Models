#!/usr/bin/env python3
"""
Mixed Sources Dataset Balancer for AI vs Human Text Classification

This script prepares balanced datasets from the RAID and M4GT corpora for training text classification models
to distinguish between human and AI-generated content across various domains.
"""

import json
import argparse
import random
import re
from pathlib import Path
from datasets import load_from_disk, load_dataset, concatenate_datasets
from typing import List, Tuple, Dict

def parse_arguments() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(description="Balance and prepare datasets from multiple sources for AI detection fine-tuning")
    
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the processed dataset")
    parser.add_argument("--total_samples_per_domain", type=int, default=2660,
                        help="Total number of samples per domain")
    parser.add_argument("--raid_ai_samples", type=int, default=790,
                        help="Number of AI samples from RAID per model per domain")
    parser.add_argument("--m4gt_ai_samples", type=int, default=180,
                        help="Number of AI samples from M4GT per model per domain")
    parser.add_argument("--raid_domains", nargs="+", 
                        default=["abstracts", "reddit"],
                        help="Domains to include from RAID dataset")
    parser.add_argument("--m4gt_domains", nargs="+", 
                        default=["arxiv", "reddit"],
                        help="Corresponding domains to include from M4GT dataset")
    
    return parser.parse_args()

def balance_dataset_human(raid_dataset, m4gt_dataset, raid_domains: List[str], m4gt_domains: List[str], 
                          samples_per_domain: int):
    """
    Create a balanced dataset from human-written content across specified domains from both RAID and M4GT.
    
    Args:
        raid_dataset: The source RAID dataset
        m4gt_dataset: The source M4GT dataset
        raid_domains: List of domains to include from RAID
        m4gt_domains: List of corresponding domains to include from M4GT
        samples_per_domain: Target number of samples per domain
    
    Returns:
        A balanced dataset with human content
    """
    list_of_datasets = []
    human_ds_raid = raid_dataset.filter(lambda example: example['model'] == 'human' and example['attack'] == 'none')
    
    print("Balancing human dataset across domains from RAID and M4GT:")
    for raid_domain, m4gt_domain in zip(raid_domains, m4gt_domains):
        print(f"Processing domain pair: RAID '{raid_domain}' - M4GT '{m4gt_domain}'")
        
        # Get samples from RAID
        ds_raid = human_ds_raid.filter(lambda example: example['domain'] == raid_domain.lower())
        raid_samples_count = len(ds_raid)
        print(f"  - Found {raid_samples_count} human samples from RAID for domain '{raid_domain}'")
        list_of_datasets.append(ds_raid)
        
        # Calculate how many samples we need from M4GT to reach the target
        m4gt_samples_needed = samples_per_domain - raid_samples_count
        
        if m4gt_samples_needed > 0:
            # Get samples from M4GT
            ds_m4gt = m4gt_dataset['train'].filter(
                lambda example: (example["model"] == 'human' and 
                                 example['source'] == 'm4gt' and 
                                 example['lang'] == 'en' and
                                 example['domain'] == m4gt_domain.lower())
            )
            
            m4gt_available = len(ds_m4gt)
            print(f"  - Found {m4gt_available} human samples from M4GT for domain '{m4gt_domain}'")
            
            if m4gt_available < m4gt_samples_needed:
                print(f"  - Warning: Not enough M4GT samples. Using all {m4gt_available} available samples.")
                list_of_datasets.append(ds_m4gt)
            else:
                print(f"  - Using {m4gt_samples_needed} samples from M4GT")
                list_of_datasets.append(ds_m4gt.select(range(m4gt_samples_needed)))
        
        print("------")
    
    return concatenate_datasets(list_of_datasets)

def balance_dataset_ai(raid_dataset, m4gt_dataset, raid_domains: List[str], m4gt_domains: List[str],
                       raid_samples_per_model: int, m4gt_samples_per_model: int):
    """
    Create a balanced dataset from AI-generated content across specified domains from both RAID and M4GT.
    
    Args:
        raid_dataset: The source RAID dataset
        m4gt_dataset: The source M4GT dataset
        raid_domains: List of domains to include from RAID
        m4gt_domains: List of corresponding domains to include from M4GT
        raid_samples_per_model: Number of samples per AI model per domain from RAID
        m4gt_samples_per_model: Number of samples per AI model per domain from M4GT
    
    Returns:
        A balanced dataset with AI-generated content
    """
    list_of_datasets = []
    ai_ds_raid = raid_dataset.filter(lambda example: example["model"] != 'human' and example['attack'] == 'none')
    
    print("Balancing AI dataset across domains and models from RAID and M4GT:")
    
    # Define models to extract with their filtering conditions
    raid_models = [
        {
            "name": "llama-chat",
            "filter": lambda ex, domain: (
                ex['model'] == 'llama-chat' and ex['domain'] == domain.lower() and
                ex['decoding'] == 'sampling' and ex['repetition_penalty'] == 'yes'
            )
        },
        {
            "name": "mistral",
            "filter": lambda ex, domain: (
                ex['model'] == 'mistral' and ex['domain'] == domain.lower() and
                ex['decoding'] == 'sampling' and ex['repetition_penalty'] == 'yes'
            )
        }
    ]
    
    m4gt_models = [
        {
            "name": "gpt4",
            "filter": lambda ex, domain: (
                ex['model'] == 'gpt4' and ex['source'] == 'm4gt' and 
                ex['lang'] == 'en' and ex['domain'] == domain.lower()
            )
        },
        {
            "name": "mixtral-8x7b",
            "filter": lambda ex, domain: (
                ex['model'] == 'mixtral-8x7b' and ex['source'] == 'm4gt' and 
                ex['lang'] == 'en' and ex['domain'] == domain.lower()
            )
        },
        {
            "name": "gemma-7b-it",
            "filter": lambda ex, domain: (
                ex['model'] == 'gemma-7b-it' and ex['source'] == 'm4gt' and 
                ex['lang'] == 'en' and ex['domain'] == domain.lower()
            )
        },
        {
            "name": "cohere",
            "filter": lambda ex, domain: (
                ex['model'] == 'cohere' and ex['source'] == 'm4gt' and 
                ex['lang'] == 'en' and ex['domain'] == domain.lower()
            )
        },
        {
            "name": "davinci",
            "filter": lambda ex, domain: (
                ex['model'] == 'davinci' and ex['source'] == 'm4gt' and 
                ex['lang'] == 'en' and ex['domain'] == domain.lower()
            )
        },
        {
            "name": "llama3-8b",
            "filter": lambda ex, domain: (
                ex['model'] == 'llama3-8b' and ex['source'] == 'm4gt' and 
                ex['lang'] == 'en' and ex['domain'] == domain.lower()
            )
        }
    ]
    
    for raid_domain, m4gt_domain in zip(raid_domains, m4gt_domains):
        print(f"Processing domain pair: RAID '{raid_domain}' - M4GT '{m4gt_domain}'")
        
        # Process RAID models
        for model_config in raid_models:
            filtered_dataset = ai_ds_raid.filter(
                lambda example: model_config["filter"](example, raid_domain)
            )
            
            available_samples = len(filtered_dataset)
            print(f"  - Found {available_samples} samples from RAID model '{model_config['name']}' in domain '{raid_domain}'")
            
            if available_samples < raid_samples_per_model:
                print(f"  - Warning: Not enough samples. Using all {available_samples} available samples.")
                list_of_datasets.append(filtered_dataset)
            else:
                print(f"  - Using {raid_samples_per_model} samples")
                list_of_datasets.append(filtered_dataset.select(range(raid_samples_per_model)))
        
        # Process M4GT models
        for model_config in m4gt_models:
            filtered_dataset = m4gt_dataset['train'].filter(
                lambda example: model_config["filter"](example, m4gt_domain)
            )
            
            available_samples = len(filtered_dataset)
            print(f"  - Found {available_samples} samples from M4GT model '{model_config['name']}' in domain '{m4gt_domain}'")
            
            if available_samples < m4gt_samples_per_model:
                print(f"  - Warning: Not enough samples. Using all {available_samples} available samples.")
                list_of_datasets.append(filtered_dataset)
            else:
                print(f"  - Using {m4gt_samples_per_model} samples")
                list_of_datasets.append(filtered_dataset.select(range(m4gt_samples_per_model)))
        
        print("------")
    
    return concatenate_datasets(list_of_datasets)

def get_balanced_datasets(raid_dataset, m4gt_dataset, 
                         raid_domains: List[str], m4gt_domains: List[str],
                         total_samples_per_domain: int,
                         raid_ai_samples: int, m4gt_ai_samples: int) -> Tuple:
    """
    Create balanced datasets for both human and AI-generated content from multiple sources.
    
    Args:
        raid_dataset: The source RAID dataset
        m4gt_dataset: The source M4GT dataset
        raid_domains: List of domains to include from RAID
        m4gt_domains: List of corresponding domains to include from M4GT
        total_samples_per_domain: Target total samples per domain
        raid_ai_samples: Number of AI samples per model per domain from RAID
        m4gt_ai_samples: Number of AI samples per model per domain from M4GT
    
    Returns:
        Tuple containing (human_dataset, ai_dataset)
    """
    raid_filtered_human = balance_dataset_human(
        raid_dataset, m4gt_dataset, 
        raid_domains, m4gt_domains, 
        total_samples_per_domain
    )
    
    raid_filtered_ai = balance_dataset_ai(
        raid_dataset, m4gt_dataset,
        raid_domains, m4gt_domains,
        raid_ai_samples, m4gt_ai_samples
    )
    
    # Calculate average text length for both datasets
    if len(raid_filtered_human) > 0:
        human_avg_length = sum(len(example['generation'].split()) for example in raid_filtered_human) / len(raid_filtered_human)
        print(f"Human dataset average length: {human_avg_length:.2f} words")
    
    if len(raid_filtered_ai) > 0:
        ai_avg_length = sum(len(example['generation'].split()) for example in raid_filtered_ai) / len(raid_filtered_ai)
        print(f"AI dataset average length: {ai_avg_length:.2f} words")
    
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
    
    print(f"Total training samples: {len(training_set)}")
    
    return training_set

def main():
    """Main function to execute the dataset balancing and preparation workflow."""
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the RAID dataset
    raid_train = load_dataset('MarcoMurgia97/raid_training_set')['train']
    
    # Load the M4GT dataset
    print(f"Loading M4GT dataset from HuggingFace")
    m4gt = load_dataset("Jinyan1/COLING_2025_MGT_en", trust_remote_code=True)
    
    # Rename columns in M4GT to match RAID format
    m4gt['train'] = m4gt['train'].rename_columns({"text": "generation", "sub_source": "domain"})
    
    # Ensure domains lists have same length
    if len(args.raid_domains) != len(args.m4gt_domains):
        raise ValueError("The number of RAID domains must match the number of M4GT domains")
    
    # Get balanced datasets
    print(f"Creating balanced datasets with {args.total_samples_per_domain} samples per domain")
    print(f"Using {args.raid_ai_samples} AI samples per model from RAID and {args.m4gt_ai_samples} from M4GT")
    
    dataset_human, dataset_ai = get_balanced_datasets(
        raid_train, m4gt,
        args.raid_domains, args.m4gt_domains,
        args.total_samples_per_domain,
        args.raid_ai_samples, args.m4gt_ai_samples
    )
    
    # Calculate total samples
    total_human = len(dataset_human)
    total_ai = len(dataset_ai)
    domains_count = len(args.raid_domains)
    
    print(f"Total human samples: {total_human} (avg {total_human/domains_count:.1f} per domain)")
    print(f"Total AI samples: {total_ai} (avg {total_ai/domains_count:.1f} per domain)")
    
    # Save the combined dataset to disk
    combined_dataset = concatenate_datasets([dataset_human, dataset_ai])
    dataset_path = output_dir / f"mixed_dataset_{'-'.join(args.raid_domains)}"
    combined_dataset.save_to_disk(dataset_path)
    print(f"Saved combined dataset to {dataset_path}")
    
    # Format the training set
    training_set = format_training_set(dataset_human, dataset_ai)
    
    # Define a descriptive name for the JSON file
    domains_str = "_".join(args.raid_domains)
    total_samples = len(training_set) // 1000
    
    json_filename = f"dataset_for_tuning_{total_samples}k_samples_{domains_str}_sourceMix.json"
    json_path = output_dir / json_filename
    
    # Save the formatted training set as JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(training_set, f, indent=2, ensure_ascii=False)
    
    print(f"Saved formatted training set to {json_path}")

if __name__ == '__main__':
    main()