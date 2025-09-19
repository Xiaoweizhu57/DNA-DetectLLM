[README.md](https://github.com/user-attachments/files/22420183/README.md)
# DNA-DetectLLM

This repository contains the implementation of **DNA-DetectLLM**, a zero-shot method for detecting AI-generated text via a mutation-repair paradigm inspired by DNA.

## 🛠️ Installation

To install the required dependencies, please run:

```bash
pip install -r requirements.txt
```

## 🔧 Code Structure

- `DNA-DetectLLM/main.py`  
  This is the entry point for running test examples of our detection algorithm. It demonstrates how to apply DNA-DetectLLM to text inputs.  
  **Note:** You need to manually configure the paths to the language model files before running. This includes models such as `falcon-7b` and `falcon-7b-instruct`.

## 📁 Data

The `Data/` directory includes two types of data used in our experiments:

- **Collected data (ours)**:  
  - 4,800 human-written texts sampled from three representative writing tasks:
    - News article writing (XSum)
    - Story generation (WritingPrompts)
    - Academic writing (Arxiv)
  - For each human-written text, we construct task-specific prompts and generate corresponding AI outputs using three large language models:
    - GPT-4 Turbo  
    - Gemini-2.0 Flash  
    - Claude-3.7 Sonnet

- **Benchmarked data (sampled)**:  
  - 2,000 balanced examples are sampled from each of the following public high-quality detection benchmarks:
    - M4  
    - DetectRL  
    - RealDet  

- **Adversarial samples**:  
  Texts modified by common adversarial attacks including insertion, deletion, substitution, and paraphrasing.


## 🚀 Evaluation

To evaluate DNA-DetectLLM’s performance on your own data, use the following command:

```bash
python eval.py --human_file=path_to_human_text.json --ai_file=path_to_ai_text.json
```

- Input format:  
  - `human_file` and `ai_file` must be JSON files with the following structure:
    ```json
    {
      "human_text": [...],
      "machine_text": [...]
    }
    ```
- Output:  
  - AUROC score and best F1 score of DNA-DetectLLM on the given dataset.

## 📄 License

This repository is released for academic research purposes only.
