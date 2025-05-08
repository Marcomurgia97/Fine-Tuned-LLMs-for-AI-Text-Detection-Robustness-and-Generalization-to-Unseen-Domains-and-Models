from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import random, re

def process_list(string_list):
    l_of_pair = []
    for i in range(0, len(string_list), 2):
        if i + 1 >= len(string_list):
            pair = string_list[i]
            l_of_pair.append(pair)
        else:
            pair = string_list[i] + string_list[i + 1]
            l_of_pair.append(pair)
    return l_of_pair

class ParaphraseAttack:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to('cuda')

    def attack(self, text, paraphrase_percentage = 50, num_beams=5, num_beam_groups=5, num_return_sequences=5, repetition_penalty=10.0, diversity_penalty=3.0, no_repeat_ngram_size=2, temperature=0.7,  max_length=256):
        '''paraphrased = ''
        text_splitted = text.split('.')
        string_concats = process_list(text_splitted)
        for sentence in string_concats:
            input_ids = self.tokenizer(
                f'paraphrase: {sentence}',
                return_tensors="pt", padding="longest",
                max_length=max_length,
                truncation=True,
            ).input_ids.to('cuda')

            outputs = self.model.generate(
                input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
                num_beams=num_beams, num_beam_groups=num_beam_groups,
                max_length=max_length, diversity_penalty=diversity_penalty
            )

            res = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            paraphrased += res[0]
        print(text)
        print(paraphrased)
        return paraphrased'''
        paraphrased = ''
        text_splitted = text.split('.')
        num_sentences_to_paraphrase = int(len(text_splitted) * (paraphrase_percentage / 100))

        # Scegli casualmente le frasi da parafrasare
        sentences_to_paraphrase_indices = random.sample(range(len(text_splitted)), num_sentences_to_paraphrase)

        for i, sentence in enumerate(text_splitted):
            if i in sentences_to_paraphrase_indices:
                input_ids = self.tokenizer(
                    f'paraphrase: {sentence}',
                    return_tensors="pt", padding="longest",
                    max_length=max_length,
                    truncation=True,
                ).input_ids.to('cuda')

                outputs = self.model.generate(
                    input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
                    num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
                    num_beams=num_beams, num_beam_groups=num_beam_groups,
                    max_length=max_length, diversity_penalty=diversity_penalty
                )

                res = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                paraphrased += ' ' + res[0]
            else:
                paraphrased += sentence + '.'

        paraphrased = paraphrased.strip().replace('  ', ' ')
        paraphrased = paraphrased.replace('!.', '!')
        paraphrased = paraphrased.replace('?.', '?')
        paraphrased = paraphrased.replace('..', '.')
        paraphrased = paraphrased.replace(' . ', '. ')

        print("Original Text:\n", text)
        print("\nParaphrased Text:\n", paraphrased)
        paraphrased = paraphrased.replace('Can you provide some examples?', '')
        paraphrased = re.sub(' +', ' ', paraphrased)
        print("\nParaphrased Text_cleanded:\n", paraphrased)

        return paraphrased





