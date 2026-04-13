# Word-level Text Generation using RNNs

Lab Assignment from [AI for Beginners Curriculum](https://github.com/microsoft/ai-for-beginners).

## Task

In this lab, you need to take any book, and use it as a dataset to train word-level text generator.

## Reference Implementation

- [Word-Level Text Generation Notebook](WordLevelGeneration.ipynb) - Uses Alice's Adventures in Wonderland from Project Gutenberg

## The Dataset

You are welcome to use any book. You can find a lot of free texts at [Project Gutenberg](https://www.gutenberg.org/), for example, here is a direct link to [Alice's Adventures in Wonderland](https://www.gutenberg.org/files/11/11-0.txt)) by Lewis Carroll.

## Key Concepts

### Word-Level vs Character-Level

| Aspect | Character-Level | Word-Level |
|--------|----------------|------------|
| Vocabulary Size | Small (~100) | Large (thousands+) |
| Sequence Length | Long | Shorter |
| Coherence | Can struggle with word formation | Better semantic coherence |

### Temperature Parameter

- **Low temperature (0.3-0.5)**: More conservative, sticks to high-probability words
- **Medium temperature (0.7-1.0)**: Balanced creativity and coherence
- **High temperature (1.5+)**: More random, may produce nonsensical output

