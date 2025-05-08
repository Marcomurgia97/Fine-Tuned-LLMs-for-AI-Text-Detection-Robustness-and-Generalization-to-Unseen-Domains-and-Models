import os
import re
import time
import argparse
from typing import List, Dict, Any

import anthropic
import datasets
from datasets import load_dataset, Dataset
import google.generativeai as genai
from openai import OpenAI

def parse_arguments() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(description="Generate datasets for AI text detection")
    
    # Dataset configuration
    parser.add_argument("--domain", type=str, required=True, choices=["medicine", "finance", "reviews"],
                        help="Domain for the dataset generation")
    parser.add_argument("--human_samples", type=int, default=150,
                        help="Number of human samples in the dataset")
    parser.add_argument("--model_samples", type=int, default=30,
                        help="Number of samples per AI model")
    parser.add_argument("--repetition_penalty", action="store_true",
                        help="Apply repetition penalty for sampling")
    parser.add_argument("--output_path", type=str, default="./generated_dataset",
                        help="Path to save the generated dataset")
    
    # Model selection
    parser.add_argument("--models", nargs="+", default=[
        "gemini-2.0-flash-exp", 
        "gemini-1.5-flash", 
        "gemini-1.5-flash-8b-latest", 
        "deepseek-ai/DeepSeek-V3", 
        "Claude-Sonnet"
    ], help="List of models to use for generation")
    
    return parser.parse_args()

def truncate_string(text: str) -> str:
    """
    Truncate the given text at the last sentence boundary (period).
    
    Args:
        text (str): Text to truncate
    
    Returns:
        str: Truncated text ending with a complete sentence
    """
    last_point = text.rfind('.')
    if last_point != -1:
        return text[:last_point + 1]
    else:
        return text

def clean_text(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean text by removing extra whitespace.
    
    Args:
        example (Dict[str, Any]): Example containing a 'generation' field
    
    Returns:
        Dict[str, Any]: Example with cleaned generation text
    """
    text = example['generation']
    if not isinstance(text, str):
        return {'generation': None}
    return {'generation': re.sub(r'\s+', ' ', text)}

def load_data(domains: List[str], human_samples: int) -> Dataset:
    """
    Load and prepare the dataset based on the specified domain.
    
    Args:
        domains (List[str]): List of domains to include
        human_samples (int): Number of human samples to select
    
    Returns:
        Dataset: Prepared dataset
    """
    transformed_data = []
    half_samples = human_samples // 2
    
    # Generic domains
    if all(domain not in domains for domain in ['reviews', 'finance', 'medicine']):
        data = load_dataset("json", data_files=f"sources/{domains[0]}_cohere.jsonl")['train'].remove_columns(
            ['source', 'source_ID'])
        data = data.filter(lambda x: (x['human_text'] is not None))
        print(f"Initial dataset size: {len(data)}")
        data = data.filter(lambda x: (len(x['human_text'].split()) < 500))
        print(f"Filtered dataset size: {len(data)}")
        
        for item in data:
            prompt = item['prompt']
            human_text = item['human_text']
            machine_text = item['machine_text']
            
            if 'abstracts' in domains:
                prompt = prompt.replace('150-220-word', '180-240-word')
                
            transformed_data.append({
                'prompt': prompt,
                'generation': human_text,
                'model': 'human',
                'attack': 'none',
                'domain': domains[0]
            })
            transformed_data.append({
                'prompt': prompt,
                'generation': machine_text,
                'model': 'gpt-3.5-turbo',
                'attack': 'none',
                'domain': domains[0]
            })
    
    # Finance or Medicine domain
    elif 'finance' in domains or 'medicine' in domains:
        data = load_dataset("Hello-SimpleAI/HC3", trust_remote_code=True, name="all")
        data = data.filter(lambda x: (x['source'] == domains[0]) and len(x['human_answers'][0].split()) > 120)
        
        filtered_human = data['train'].map(
            lambda x: {'model': 'human', 'text': x['human_answers'], 'prompt': x['question']}
        ).select(range(0, half_samples))
        
        filtered_ai = data['train'].map(
            lambda x: {'model': 'machine', 'text': x['chatgpt_answers'], 'prompt': x['question']}
        ).select(range(0, half_samples))
        
        common_questions = len(set(filtered_human['question']).intersection(filtered_ai['question']))
        print(f"Number of common questions: {common_questions}")
        
        data = datasets.concatenate_datasets([filtered_human, filtered_ai])
        
        for item in data:
            word_requirement = "200" if 'finance' in domains else "120"
            transformed_data.append({
                'prompt': f"{item['question']}. The text must be at least {word_requirement} words",
                'generation': item['text'][0],
                'model': 'human' if item['model'] == 'human' else 'machine',
                'attack': 'none',
                'domain': domains[0]
            })
    
    # Reviews domain
    elif 'reviews' in domains:
        data = load_dataset('MarcoMurgia97/raid_training_set')['train']
        data = data.filter(lambda x: (x['domain'] == domains[0] and x['attack'] == 'none'))
        
        data_h = data.filter(lambda x: (x['model'] == 'human'))
        data_ai = data.filter(lambda x: ('chat'.lower() in x['model'].lower()))
        
        human_titles = set(item['title'] for item in data_h)
        ai_titles = set(item['title'] for item in data_ai)
        
        # Find common unique titles
        common_titles = list(human_titles & ai_titles)
        # Take exactly n_elements unique titles
        selected_titles = common_titles[:150]
        
        # Create dictionaries for quick lookup
        human_dict = {item['title']: item for item in data_h}
        ai_dict = {item['title']: item for item in data_ai}
        
        # Create new lists with exactly one item per title
        matched_human_data = [human_dict[title] for title in selected_titles]
        matched_ai_data = [ai_dict[title] for title in selected_titles]
        
        # Convert to datasets
        matched_human_dataset = datasets.Dataset.from_list(matched_human_data)
        matched_ai_dataset = datasets.Dataset.from_list(matched_ai_data)
        
        common_titles_count = len(set(matched_human_dataset['title']).intersection(set(matched_ai_dataset['title'])))
        print(f"Number of common titles: {common_titles_count}")
        
        data = datasets.concatenate_datasets([matched_human_dataset, matched_ai_dataset])
        
        for item in data:
            transformed_data.append({
                'prompt': f"{item['prompt']}. The text must be at least 200 words",
                'generation': item['generation'],
                'model': 'human' if item['model'] == 'human' else 'machine',
                'attack': item['attack'],
                'domain': domains[0]
            })
    
    # Create the Hugging Face dataset
    tmp_dataset = datasets.Dataset.from_list(transformed_data)
    return tmp_dataset

def generate_text_gemini(prompt: str, decoding: str, model_name: str) -> str:
    """
    Generate text using the Gemini model.
    
    Args:
        prompt (str): Prompt for text generation
        decoding (str): Decoding method ('greedy' or 'sampling')
        model_name (str): Name of the Gemini model to use
    
    Returns:
        str: Generated text
    """
    prompt = f'{prompt}. Only answer the request, without phrases like 'sure, here\'s...\' or similar phrases'
    
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(model_name)
    
    # Set generation parameters based on decoding method
    temperature = 0.0 if decoding == 'greedy' else 1.0
    top_p = 0.0 if decoding == 'greedy' else 1.0
    frequency_penalty = 0.0 if decoding == 'greedy' else 0.5
    presence_penalty = 0.0 if decoding == 'greedy' else 0.0
    
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            max_output_tokens=512,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        ),
    )
    
    # Rate limiting
    if '2.0' in model_name:
        time.sleep(5)
    else:
        time.sleep(3)
    
    answer = response.text
    answer = re.sub(r'\s+', ' ', answer)
    
    # Do not truncate for recipes domain
    if 'recipes' not in [args.domain]:
        answer = truncate_string(answer)
        
    print(f"Generated {len(answer.split())} words with {model_name} using {decoding} decoding")
    return answer

def generate_text_claude(prompt: str, decoding: str) -> str:
    """
    Generate text using the Claude model.
    
    Args:
        prompt (str): Prompt for text generation
        decoding (str): Decoding method ('greedy' or 'sampling')
    
    Returns:
        str: Generated text
    """
    prompt = f'{prompt}. Only answer the request, without phrases like 'sure, here\'s...\' or similar phrases'
    
    client = anthropic.Anthropic(
        api_key=os.environ["CLAUDE_API_KEY"],
    )
    
    # Set generation parameters based on decoding method
    temperature = 0.0 if decoding == 'greedy' else 1.0
    top_p = 0.0 if decoding == 'greedy' else 1.0
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=512,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        top_p=top_p,
    )
    
    answer = message.content[0].text
    answer = re.sub(r'\s+', ' ', answer)
    
    # Do not truncate for recipes domain
    if 'recipes' not in [args.domain]:
        answer = truncate_string(answer)
        
    print(f"Generated {len(answer.split())} words with Claude using {decoding} decoding")
    return answer

def generate_text_deepinfra(prompt: str, decoding: str, model_name: str) -> str:
    """
    Generate text using models hosted on DeepInfra.
    
    Args:
        prompt (str): Prompt for text generation
        decoding (str): Decoding method ('greedy' or 'sampling')
        model_name (str): Name of the model to use
    
    Returns:
        str: Generated text
    """
    prompt = f'{prompt}. Answer only the request, without phrases such as \'sure, here\'s...\' or similar'
    
    client = OpenAI(
        api_key=os.environ["DEEPINFRA_API_KEY"],
        base_url="https://api.deepinfra.com/v1/openai",
    )
    
    # Set generation parameters based on decoding method
    temperature = 0.0 if decoding == 'greedy' else 1.0
    frequency_penalty = 0.0 if decoding == 'greedy' else 0.5
    presence_penalty = 0.0 if decoding == 'greedy' else 0.0
    top_p = 0.1 if decoding == 'greedy' else 1.0
    
    # Extract model provider and name
    model_parts = model_name.split('/')
    full_model_name = f"{model_parts[0]}/{model_parts[1]}" if len(model_parts) > 1 else model_name
    
    chat_completion = client.chat.completions.create(
        model=full_model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        top_p=top_p,
    )
    
    response = chat_completion.choices[0].message.content
    response = re.sub(r'\s+', ' ', response)
    
    # Do not truncate for recipes domain
    if 'recipes' not in [args.domain]:
        response = truncate_string(response)
        
    print(f"Generated {len(response.split())} words with {model_name} using {decoding} decoding")
    return response

def generate_dataset(ds: Dataset, models: List[str], human_samples: int, model_samples: int, 
                     rep_penalty: bool) -> Dataset:
    """
    Generate a combined dataset with human samples and AI-generated samples.
    
    Args:
        ds (Dataset): Source dataset to generate from
        models (List[str]): List of AI models to use
        human_samples (int): Number of human samples to include
        model_samples (int): Number of samples per AI model
        rep_penalty (bool): Whether to apply repetition penalty
    
    Returns:
        Dataset: Combined dataset with all samples
    """
    # Filter human samples
    ds_human = ds.filter(lambda example: example['model'] == 'human').select(range(human_samples))
    
    # Filter AI samples to use as prompts
    ds_ai = ds.filter(lambda example: example['model'] != 'human').select(range(human_samples))
    
    # Clean human text samples
    ds_human = ds_human.map(clean_text)
    
    # Initialize list to store all datasets
    list_of_ds = [ds_human]
    
    # For each model, generate text with different decoding methods
    for i, model_name in enumerate(models):
        start_idx = i * model_samples
        end_idx = start_idx + model_samples
        half_samples = model_samples // 2
        
        # Handle different model types
        if not model_name.startswith('Claude') and not model_name.startswith('gemini'):
            # Generate with greedy decoding
            greedy_samples = ds_ai.select(range(start_idx, start_idx + half_samples)).map(
                lambda x: {
                    'generation': generate_text_deepinfra(x['prompt'], 'greedy', model_name),
                    'model': model_name.split('/')[1] if '/' in model_name else model_name,
                    'decoding': 'greedy',
                    'repetition_penalty': 'no'
                }
            )
            list_of_ds.append(greedy_samples)
            
            # Generate with sampling decoding
            sampling_samples = ds_ai.select(range(start_idx + half_samples, end_idx)).map(
                lambda x: {
                    'generation': generate_text_deepinfra(x['prompt'], 'sampling', model_name),
                    'model': model_name.split('/')[1] if '/' in model_name else model_name,
                    'decoding': 'sampling',
                    'repetition_penalty': 'yes' if rep_penalty else 'no'
                }
            )
            list_of_ds.append(sampling_samples)
            
        elif model_name.startswith('Claude'):
            # Generate with greedy decoding
            greedy_samples = ds_ai.select(range(start_idx, start_idx + half_samples)).map(
                lambda x: {
                    'generation': generate_text_claude(x['prompt'], 'greedy'),
                    'model': 'Claude-sonnet',
                    'decoding': 'greedy',
                    'repetition_penalty': 'no'
                }
            )
            list_of_ds.append(greedy_samples)
            
            # Generate with sampling decoding
            sampling_samples = ds_ai.select(range(start_idx + half_samples, end_idx)).map(
                lambda x: {
                    'generation': generate_text_claude(x['prompt'], 'sampling'),
                    'model': 'Claude-sonnet',
                    'decoding': 'sampling',
                    'repetition_penalty': 'no'
                }
            )
            list_of_ds.append(sampling_samples)
            
        elif model_name.startswith('gemini'):
            # Generate with greedy decoding
            greedy_samples = ds_ai.select(range(start_idx, start_idx + half_samples)).map(
                lambda x: {
                    'generation': generate_text_gemini(x['prompt'], 'greedy', model_name),
                    'model': model_name,
                    'decoding': 'greedy',
                    'repetition_penalty': 'no'
                }
            )
            list_of_ds.append(greedy_samples)
            
            # Generate with sampling decoding
            sampling_samples = ds_ai.select(range(start_idx + half_samples, end_idx)).map(
                lambda x: {
                    'generation': generate_text_gemini(x['prompt'], 'sampling', model_name),
                    'model': model_name,
                    'decoding': 'sampling',
                    'repetition_penalty': 'yes' if rep_penalty else 'no'
                }
            )
            list_of_ds.append(sampling_samples)
    
    # Combine all datasets
    return datasets.concatenate_datasets(list_of_ds)

def main():
    """
    Main function to orchestrate the dataset generation process.
    """
    global args
    args = parse_arguments()
    
    # Load and prepare the dataset
    print(f"Loading data for domain: {args.domain}")
    dataset = load_data([args.domain], args.human_samples)
    
    # Generate combined dataset
    print(f"Generating dataset with models: {args.models}")
    dataset = generate_dataset(
        dataset,
        args.models,
        args.human_samples,
        args.model_samples,
        args.repetition_penalty
    )
    
    # Prepare output path
    rep_suffix = '_rep_pen' if args.repetition_penalty else ''
    output_path = os.path.join(
        args.output_path, 
        f"test_set_{args.domain}{rep_suffix}_{args.human_samples}_{args.model_samples}"
    )
    
    # Save the dataset
    print(f"Saving dataset to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset.save_to_disk(output_path)
    print(f"Dataset generation complete. Dataset info:\n{dataset}")

if __name__ == "__main__":
    main()