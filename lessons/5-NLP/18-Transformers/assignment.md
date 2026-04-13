# Assignment: Transformers

Experiment with Transformers on HuggingFace! Try some of the scripts they provide to work with the various models available on their site: https://huggingface.co/docs/transformers/run_scripts. Try one of their datasets, then import one of your own from this curriculum or from Kaggle and see if you can generate interesting texts. Produce a notebook with your findings.

## Reference Implementation

- [HuggingFace Experiments Notebook](HuggingFace_Experiments.ipynb) - Demonstrates multiple NLP tasks:
  - Sentiment Analysis
  - Named Entity Recognition
  - Text Generation (GPT-2)
  - Question Answering
  - Text Summarization
  - Translation
  - Zero-Shot Classification
  - AG News Classification

## Tasks to Explore

1. **Text Classification**: Fine-tune BERT on a custom dataset (e.g., sentiment analysis)
2. **Named Entity Recognition**: Extract entities from news articles
3. **Text Generation**: Experiment with GPT-2 for creative writing
4. **Question Answering**: Build a QA system for your documents
5. **Translation**: Translate between multiple languages

## Useful Models

| Task | Recommended Model |
|------|------------------|
| Classification | `bert-base-uncased`, `distilbert-base-uncased` |
| Generation | `gpt2`, `gpt2-medium`, `gpt2-large` |
| QA | `distilbert-base-cased-distilled-squad` |
| Summarization | `facebook/bart-large-cnn`, `t5-base` |
| Translation | `t5-base`, ` Helsinki-NLP/opus-mt-*` |