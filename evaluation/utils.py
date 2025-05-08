import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM
from collections import Counter, defaultdict


def load_model_tokenizer(model_path):
    '''tok = AutoTokenizer.from_pretrained(model_path)

    mod = AutoPeftModelForCausalLM.from_pretrained(
        model_path,  # YOUR MODEL YOU USED FOR TRAINING
        load_in_4bit=True,
    )

    mod.config.use_cache = True

    return mod, tok'''

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.bos_token_id = 1
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    ).to('cuda')
    model.config.use_cache = True

    return model, tokenizer


def gen_answer(model, tokenizer, prompt, th):
    messages = [
        {"role": "user", "content": prompt},
    ]
    toReturn = ''
    prob = 0.0
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    model_inputs = encodeds.to('cuda')

    generated_ids = model.generate(model_inputs, max_new_tokens=1, do_sample=False, temperature=0.0,
                                   eos_token_id=tokenizer.eos_token_id,
                                   return_dict_in_generate=True,
                                   output_scores=True)
    decoded = tokenizer.batch_decode(generated_ids.sequences)
    scores = generated_ids.scores 

    for i, score in enumerate(scores):
        next_token_probability = torch.softmax(score[0, :], dim=-1)
        sorted_ids = torch.argsort(next_token_probability, dim=-1, descending=True)

        print(f"\nToken generato {i + 1}:")

        for choice_idx in range(2):
            token_id = sorted_ids[choice_idx].item()
            token_probability = next_token_probability[token_id].item()
            token_choice = f"{tokenizer.decode([token_id])}({100 * token_probability:.2f}%)"
            print(f"Choice {choice_idx + 1}: {token_choice}")
            if th is not None:
                if choice_idx == 0 and tokenizer.decode([token_id]).lower() == 'machine':
                    if token_probability > th:
                        return 'machine'
                    else:
                        return 'human'
                else:
                    return 'human'
            else:
                if choice_idx == 0 and 'machine' in tokenizer.decode([token_id]).lower():
                    prob = token_probability
                elif choice_idx == 0 and 'machine' not in tokenizer.decode([token_id]).lower():
                    prob = 1.0 - token_probability
                return decoded[0].split("<|end_header_id|>", 3)[-1], prob

