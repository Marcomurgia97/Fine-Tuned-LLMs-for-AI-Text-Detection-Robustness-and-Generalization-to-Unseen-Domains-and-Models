from tqdm import tqdm

from gltr_model import LM


class GLTR:

    RANK = 10

    def __init__(self) -> None:
        self.model = LM()

    def inference(self, text: str) -> float:
        predictions = []
        results = 0.0
        try:
            results = self.model.check_probabilities(text, topk=1)
        except IndexError as e:
            print(f"Error: {e}")
            predictions = 1  # Machine

        numbers = [item[0] for item in results["real_topk"]]
        numbers = [item <= self.RANK for item in numbers]

        predictions = sum(numbers) / len(numbers)

        return predictions

    @classmethod
    def set_param(cls, rank: int, prob: float) -> None:
        cls.RANK = rank


