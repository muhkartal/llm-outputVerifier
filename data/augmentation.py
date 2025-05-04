import random
from typing import Dict, List, Tuple, Union

import numpy as np
from datasets import Dataset
import re


def parse_gsm8k_reasoning(
    example: Dict[str, str]
) -> Tuple[List[str], List[int]]:
    reasoning = example["answer"]

    steps = re.split(r"\n", reasoning)
    steps = [step for step in steps if step.strip()]

    labels = [0] * len(steps)  # 0 = grounded, 1 = hallucinated

    return steps, labels


def fact_distortion(
    step: str
) -> str:
    modifiers = [
        ("increase", "decrease"),
        ("more", "less"),
        ("larger", "smaller"),
        ("add", "subtract"),
        ("multiply", "divide"),
        ("twice", "half"),
        ("double", "halve"),
    ]

    for original, replacement in modifiers:
        if original in step.lower():
            return step.lower().replace(original, replacement)

    return step


def logical_error(
    step: str
) -> str:
    operations = {
        r"(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)": lambda m: f"{m.group(1)} + {m.group(2)} = {int(m.group(1)) + int(m.group(2)) + random.randint(1, 5)}",
        r"(\d+)\s*-\s*(\d+)\s*=\s*(\d+)": lambda m: f"{m.group(1)} - {m.group(2)} = {int(m.group(1)) - int(m.group(2)) + random.randint(1, 5)}",
        r"(\d+)\s*\*\s*(\d+)\s*=\s*(\d+)": lambda m: f"{m.group(1)} * {m.group(2)} = {int(m.group(1)) * int(m.group(2)) + random.randint(1, 10)}",
        r"(\d+)\s*/\s*(\d+)\s*=\s*(\d+)": lambda m: f"{m.group(1)} / {m.group(2)} = {int(m.group(1)) // int(m.group(2)) + random.randint(1, 3)}",
    }

    for pattern, replacement_func in operations.items():
        match = re.search(pattern, step)
        if match:
            return re.sub(pattern, replacement_func(match), step)

    return step


def number_substitution(
    step: str
) -> str:
    numbers = re.findall(r"\d+", step)
    if not numbers:
        return step

    number_to_replace = random.choice(numbers)
    new_number = str(int(number_to_replace) + random.randint(1, 10))

    return step.replace(number_to_replace, new_number, 1)


def create_synthetic_hallucinations(
    dataset: Dataset,
    corruption_rate: float = 0.3,
    corruption_types: List[str] = None,
) -> Dataset:
    if corruption_types is None:
        corruption_types = ["fact_distortion", "logical_error", "number_substitution"]

    corruption_funcs = {
        "fact_distortion": fact_distortion,
        "logical_error": logical_error,
        "number_substitution": number_substitution,
    }

    def process_example(example: Dict[str, str]) -> Dict[str, Union[List[str], List[int]]]:
        reasoning_steps, hallucination_labels = parse_gsm8k_reasoning(example)

        for i in range(len(reasoning_steps)):
            if random.random() < corruption_rate:
                corruption_type = random.choice(corruption_types)
                corruption_func = corruption_funcs[corruption_type]

                original_step = reasoning_steps[i]
                corrupted_step = corruption_func(original_step)

                if corrupted_step != original_step:
                    reasoning_steps[i] = corrupted_step
                    hallucination_labels[i] = 1  # Mark as hallucinated

        return {
            "reasoning_steps": reasoning_steps,
            "hallucination_labels": hallucination_labels,
            "question": example["question"],
            "original_answer": example["answer"],
        }

    return dataset.map(process_example)
