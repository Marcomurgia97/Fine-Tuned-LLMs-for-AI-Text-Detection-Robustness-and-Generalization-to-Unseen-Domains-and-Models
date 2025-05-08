# Fine Tuned LLMs for AI-Text-Detection Robustness and Generalization to Unseen Domain and Models

[![Python Version](https://img.shields.io/badge/python-3.9.13-blue.svg)](https://python.org) <!-- Optional: Add badges like this -->

## Description

This repository contains the source code for a study on using Large Language Models (LLMs) and SFT fine tuning to detect AI-generated versus human-generated texts.

The project explores the application of fine tuned LLMs detection systems across different domains and against adversarial attacks, such as paraphrasing and synonym substitution.

  
The results demonstrate that the proposed approach is promising and can match or exceed the performance of current state-of-the-art methods.
## Setup and Installation

Follow these steps to set up the project environment and install dependencies.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Marcomurgia97/A-novel-approach-for-AI-detection-content-based-on-fine-tuned-LLMs.git
    ```

2.  **Create and activate a virtual environment (Recommended):**
    *   On macOS and Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   On Windows:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
3.  **Install required packages:**
    Make sure your virtual environment is activated before running this command.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Here are examples of how to run the main scripts in this project.

*   **To run the AI detection benchmark evaluation:**
    ```bash
    python evaluation/evaluation.py --model_path <huggingface-model-id> --results_dir results/my_evaluation --domains wiki reddit --attacks none paraphrase --threshold 0.6 --dataset_samples 100 --setting setting_1
    ```
    This script evaluates a specified AI text detection model (`--model_path`) on benchmark datasets. It tests performance across different domains (`--domains`) and adversarial attack types (`--attacks`), using a specific decision threshold (`--threshold`) and number of samples (`--dataset_samples`). Results (precision, recall, F1, etc.) are saved in the specified directory (`--results_dir`) under the chosen setting (`--setting`). Replace `<huggingface-model-id>` with the actual model identifier (e.g., [Llama3.1-8b-SingleDomain](https://huggingface.co/MarcoMurgia97/Llama3.1-8b_SingleDomain)). You can customize the domains, attacks, threshold, sample size, setting, and output directory as needed.
In the script the test set used in the experiments described in the paper is used by default, but it can be easily changed in the “load_benchmark” function with a user-chosen dataset


## Links
**Models**
*   **[Llama3.1-8b-SingleDomain](https://huggingface.co/MarcoMurgia97/Llama3.1-8b_SingleDomain)** - SFT training on one domain (abstracts)
*   **[Llama3.1-8b-DualDomain](https://huggingface.co/MarcoMurgia97/Llama3.1-8b_DualDomain)** - SFT training on two domains (abstracts and reddit)

**RAID subset**
*   **[Subset](https://huggingface.co/datasets/MarcoMurgia97/raid_training_set)** 

**Threshold Optimization Set**
*   **[Threshold Optimization Set](https://huggingface.co/datasets/MarcoMurgia97/Threshold_Optimization_Set)** 

**Test Set**
*   **[Test set](https://huggingface.co/datasets/MarcoMurgia97/test_set_setting_3)** 

## Contact

* For any questions, suggestions, or issues, please feel free to open an issue on this repository or contact Marco Murgia at marco.murgia3@unica.it.

---
