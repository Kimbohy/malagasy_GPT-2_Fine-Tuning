# 🇲🇬 GPT-2 Fine-Tuning for Malagasy Text Generation

Fine-tuning of **GPT-2** on the **Soratra Malagasy text corpus** to explore automatic text generation in the Malagasy language.

The project provides a complete training pipeline covering data exploration, preprocessing, tokenization, model training, evaluation, and text generation.

## 📌 Overview

Large Language Models (LLMs) are predominantly trained on high-resource languages, while languages such as **Malagasy** remain significantly underrepresented.

This project explores whether a pre-trained GPT-2 model can learn Malagasy linguistic patterns through fine-tuning on a Malagasy corpus.

The resulting model is intended as an **experimental language model for Malagasy text generation**, rather than a production-ready conversational AI.

## ✨ Features

- 📊 Malagasy corpus exploration and statistics
- 🧹 Text cleaning and preprocessing
- 🔤 GPT-2 tokenization
- 🤗 Hugging Face `datasets` integration
- 🧠 GPT-2 causal language modeling
- ⚙️ Configurable training parameters
- 📈 Training monitoring with Weights & Biases
- 🧪 Train / validation / test split
- 📏 Test-set evaluation using language-model loss
- ✍️ Malagasy text generation
- 💾 Model and tokenizer export

## 🏗️ Pipeline

```text
             Malagasy Corpus
                    │
                    ▼
          ┌───────────────────┐
          │ Data Exploration  │
          └─────────┬─────────┘
                    │
                    ▼
          ┌───────────────────┐
          │ Data Cleaning     │
          └─────────┬─────────┘
                    │
                    ▼
          ┌───────────────────┐
          │ Train / Val / Test│
          │      Split        │
          └─────────┬─────────┘
                    │
                    ▼
          ┌───────────────────┐
          │ GPT-2 Tokenizer   │
          └─────────┬─────────┘
                    │
                    ▼
          ┌───────────────────┐
          │ GPT-2 Fine-Tuning │
          └─────────┬─────────┘
                    │
             ┌──────┴──────┐
             ▼             ▼
        Evaluation     Text Generation
             │             │
             └──────┬──────┘
                    ▼
             Fine-tuned Model
```

## 🧠 Model

The project uses the pre-trained **GPT-2** architecture as the starting point.

Instead of training a language model from scratch, GPT-2 is fine-tuned on Malagasy text so that its language modeling behavior adapts to the target corpus.

The default configuration uses:

| Parameter               |  Value |
| ----------------------- | -----: |
| Base model              | `gpt2` |
| Maximum sequence length |    256 |
| Epochs                  |      3 |
| Learning rate           | `5e-5` |
| Batch size              |      8 |
| Gradient accumulation   |      2 |
| Warmup steps            |    500 |
| Weight decay            |   0.01 |
| Train split             |    90% |
| Validation split        |     5% |
| Test split              |     5% |
| Seed                    |     42 |

All major parameters can be modified from the `CONFIG` dictionary in the notebook.

## 📚 Dataset

The model is fine-tuned on the **Soratra** corpus, a collection of Malagasy text.

The notebook expects a CSV file containing a `text` column:

```csv
text
"Ny firenena Malagasy..."
"Madagasikara dia..."
"Ny tantaran'i Madagasikara..."
```

### Expected structure

```text
soratra.csv
└── text
    ├── Malagasy text 1
    ├── Malagasy text 2
    ├── Malagasy text 3
    └── ...
```

> **Note:** Make sure you have the appropriate rights to use and redistribute the dataset. The dataset itself is not included in this repository unless explicitly stated.

## 🧹 Preprocessing

The preprocessing pipeline performs several basic operations:

- Remove missing or empty texts
- Normalize excessive whitespace
- Strip leading and trailing whitespace
- Remove HTML tags
- Remove empty samples

The cleaned corpus is then randomly shuffled and divided into training, validation, and test sets.

### Important note

The current preprocessing removes non-ASCII characters. This is potentially problematic for Malagasy text because it can remove valid characters and linguistic information.

For a more robust Malagasy language model, preprocessing should ideally preserve Unicode characters.

## 🔤 Tokenization

The project uses the GPT-2 tokenizer provided by Hugging Face Transformers.

```python
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
```

Texts are tokenized with a maximum sequence length of 256 tokens.

Because GPT-2 is a **causal language model**, the training objective is next-token prediction rather than masked-token prediction.

```text
Input:
Ny firenena Malagasy

Model learns to predict:
Ny → firenena → Malagasy → ...
```

## 🚀 Installation

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

Install the required dependencies:

```bash
pip install torch
pip install transformers
pip install datasets
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install tqdm
pip install wandb
```

Or create a `requirements.txt` and install everything with:

```bash
pip install -r requirements.txt
```

## ▶️ Usage

The project is currently provided as a Jupyter notebook.

Start Jupyter:

```bash
jupyter notebook
```

Then open:

```text
notebook.ipynb
```

Before training, update the dataset path in the configuration:

```python
CONFIG = {
    "data_path": "/path/to/soratra.csv",
    ...
}
```

For Kaggle, the default path can be:

```python
"data_path": "/kaggle/input/soratra/soratra.csv"
```

### Quick experiment

To train on a smaller subset of the dataset, set:

```python
"max_samples": 10000
```

For the complete corpus:

```python
"max_samples": None
```

## 📈 Experiment Tracking

Training experiments are monitored using **Weights & Biases**.

The project logs:

- Training loss
- Validation loss
- Test loss
- Training metrics
- Sample text generations

The experiment is initialized with:

```python
wandb.init(
    project="gpt2-soratra"
)
```

This makes it possible to monitor training progress and compare different experiments.

> **Security:** Never commit your W&B API key to Git. Use `wandb login` or an environment variable instead.

## 🧪 Evaluation

The model is evaluated on a held-out test set after training.

The primary metric currently used is:

### Test Loss

```text
test_loss
```

For a causal language model, the loss measures how well the model predicts the next tokens in unseen text.

Perplexity can also be derived from the loss:

$$
\text{Perplexity} = e^{\text{Loss}}
$$

Lower loss and perplexity generally indicate better next-token prediction on the evaluation corpus.

However, these metrics do not necessarily measure how natural or useful the generated Malagasy text is.

## ✍️ Text Generation

After training, the model can generate Malagasy text from a prompt.

Example:

```python
prompt = "Ny firenena Malagasy"

generated = generate_text(
    prompt,
    max_length=150,
    num_return_sequences=2
)
```

The generation function exposes several sampling parameters:

```python
temperature=0.8
top_k=50
top_p=0.95
```

These parameters control the diversity and randomness of generated text.

### Example prompts

```text
Hiverina ho zanatany amin'ny endriny maoderina i Madagasikara

Ny firenena Malagasy

Amin'ny vanim-potoana
```

The model attempts to continue these prompts using patterns learned from the Malagasy corpus.

## 📁 Project Structure

```text
.
├── notebook.ipynb
├── README.md
├── requirements.txt
└── ...
```

The notebook is organized into the following sections:

```text
1. Imports and Setup
2. Data Exploration
3. Data Cleaning
4. Tokenization
5. Model Initialization
6. Training Setup
7. Training
8. Evaluation
9. Text Generation
10. Summary
```

## ⚠️ Limitations

This project is an experimental exploration of Malagasy language modeling and has several limitations.

### Dataset

The quality, size, diversity, and domain distribution of the corpus directly affect the resulting model.

### Tokenizer

The original GPT-2 tokenizer was not specifically designed for Malagasy. Its subword segmentation may therefore be inefficient for Malagasy vocabulary.

### Preprocessing

The current preprocessing pipeline should be improved to properly preserve Unicode characters and Malagasy-specific linguistic information.

### Model size

The default `gpt2` model is relatively small compared with modern language models. Larger architectures may provide better generation quality, at the cost of significantly higher computational requirements.

### Evaluation

Language-model loss alone does not fully capture the quality, grammaticality, coherence, or factuality of generated Malagasy text.

Human evaluation and Malagasy-specific evaluation datasets would provide a more meaningful assessment.

## 🔬 Future Work

Several improvements could be explored:

- 🇲🇬 Build or adopt a tokenizer better adapted to Malagasy
- 📚 Increase the size and diversity of the Malagasy corpus
- 🧹 Improve Malagasy-specific preprocessing
- 🧠 Experiment with `gpt2-medium` and larger models
- ⚙️ Perform systematic hyperparameter tuning
- 📏 Report perplexity alongside loss
- 🧪 Compare different tokenizers
- 🔍 Evaluate generated text with Malagasy speakers
- 📊 Create a dedicated Malagasy language evaluation benchmark
- 🧩 Experiment with domain-specific fine-tuning
- 🤗 Publish the resulting model on Hugging Face
- 🌍 Compare the fine-tuned model with multilingual language models

## 🎯 Goal

The main goal of this project is to investigate **Malagasy language modeling with generative AI** and contribute to experimentation around NLP for a relatively low-resource language.

More broadly, it serves as a practical exploration of:

```text
Low-resource NLP
       ↓
Language Modeling
       ↓
Transfer Learning
       ↓
GPT-2 Fine-Tuning
       ↓
Malagasy Text Generation
```

## 🛠️ Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- GPT-2
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Weights & Biases
- Jupyter Notebook

---

**Made with 🇲🇬 and 🤗 for Malagasy NLP research and experimentation.**
