# Assignment: RNN for Sentiment Analysis

This assignment demonstrates RNN/LSTM for sentiment analysis using the IMDB movie reviews dataset.

**Dataset**: IMDB Movie Reviews - binary sentiment classification (positive/negative)

## Setup

```python
import torch
import torchtext
from torchtext.datasets import IMDB
import collections
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

## Load Dataset

IMDB dataset contains 50,000 movie reviews for binary sentiment classification.

```python
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

def load_imdb_dataset(vocab_size=10000):
    print("Loading IMDB dataset...")
    
    # Use torchtext 0.6.0 API
    TEXT = torchtext.data.Field(tokenize='basic_english', include_lengths=True)
    LABEL = torchtext.data.LabelField(dtype=torch.float)
    
    train_data, test_data = torchtext.datasets.IMDB.splits(TEXT, LABEL, root='./data')
    
    # Build vocabulary
    print('Building vocabulary...')
    TEXT.build_vocab(train_data, max_size=vocab_size-2)
    LABEL.build_vocab(train_data)
    
    vocab = TEXT.vocab
    
    # Convert to list format: (label, text)
    train_dataset = [(1 if label == 'pos' else 0, ' '.join(text)) for label, text in zip(train_data.label, train_data.text)]
    test_dataset = [(1 if label == 'pos' else 0, ' '.join(text)) for label, text in zip(test_data.label, test_data.text)]
    
    print(f'Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}')
    print(f'Vocabulary size: {len(vocab)}')
    
    return train_dataset, test_dataset, vocab

train_dataset, test_dataset, vocab = load_imdb_dataset()
classes = ['Negative', 'Positive']  # 0=negative, 1=positive
```

## Data Preparation

Convert text to tensors with padding for batch processing.

```python
vocab_size = len(vocab)
pad_idx = vocab.stoi['<pad>']
unk_idx = vocab.stoi['<unk>']

def encode_text(text, vocab, tokenizer):
    """Convert text to list of token indices"""
    return [vocab.stoi.get(token, unk_idx) for token in tokenizer(text)]

def collate_batch(batch, vocab, tokenizer, pad_idx):
    """Prepare batch for training"""
    labels, texts, lengths = [], [], []
    for label, text in batch:
        labels.append(label)
        encoded = encode_text(text, vocab, tokenizer)
        texts.append(encoded)
        lengths.append(len(encoded))
    
    # Pad sequences
    max_len = max(lengths)
    texts_padded = [t + [pad_idx] * (max_len - len(t)) for t in texts]
    
    return (
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(texts_padded, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long)
    )

from functools import partial
collate_fn = partial(collate_batch, vocab=vocab, tokenizer=tokenizer, pad_idx=pad_idx)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

print(f"Number of training batches: {len(train_loader)}")
print(f"Number of test batches: {len(test_loader)}")
```

## Model 1: Simple RNN Classifier

```python
class RNNClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_class, pad_idx):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = torch.nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, num_class)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, text, lengths=None):
        embedded = self.dropout(self.embedding(text))
        output, hidden = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))

model_rnn = RNNClassifier(vocab_size, embed_dim=128, hidden_dim=64, num_class=2, pad_idx=pad_idx).to(device)
print(model_rnn)
```

```python
def train_model(model, train_loader, test_loader, epochs=5, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        for labels, texts, lengths in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        train_acc = correct / total
        
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for labels, texts, lengths in test_loader:
                texts, labels = texts.to(device), labels.to(device)
                outputs = model(texts)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        val_acc = correct / total
        print(f'Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}')

print("Training Simple RNN...")
train_model(model_rnn, train_loader, test_loader, epochs=5)
```

## Model 2: LSTM Classifier

LSTM addresses the vanishing gradient problem and can learn longer dependencies.

```python
class LSTMClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_class, pad_idx, num_layers=2, bidirectional=True):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = torch.nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=0.3,
            batch_first=True
        )
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = torch.nn.Linear(fc_input_dim, num_class)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, text, lengths=None):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        # Concatenate final forward and backward hidden states
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        return self.fc(self.dropout(hidden))

model_lstm = LSTMClassifier(vocab_size, embed_dim=128, hidden_dim=64, num_class=2, pad_idx=pad_idx).to(device)
print(model_lstm)
```

```python
print("Training Bidirectional LSTM...")
train_model(model_lstm, train_loader, test_loader, epochs=5)
```

## Model 3: LSTM with Packed Sequences

Packed sequences avoid processing padding tokens, improving efficiency.

```python
class LSTMPackedClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_class, pad_idx, num_layers=2, bidirectional=True):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = torch.nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=0.3,
            batch_first=True
        )
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = torch.nn.Linear(fc_input_dim, num_class)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, text, lengths):
        embedded = self.dropout(self.embedding(text))
        # Pack padded sequence
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed)
        # Concatenate final forward and backward hidden states
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        return self.fc(self.dropout(hidden))

model_packed = LSTMPackedClassifier(vocab_size, embed_dim=128, hidden_dim=64, num_class=2, pad_idx=pad_idx).to(device)
print(model_packed)
```

```python
def train_packed_model(model, train_loader, test_loader, epochs=5, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        for labels, texts, lengths in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        train_acc = correct / total
        
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for labels, texts, lengths in test_loader:
                texts, labels = texts.to(device), labels.to(device)
                outputs = model(texts, lengths)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        val_acc = correct / total
        print(f'Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}')

print("Training LSTM with Packed Sequences...")
train_packed_model(model_packed, train_loader, test_loader, epochs=5)
```

## Comparison Summary

| Model | Key Features |
|-------|-------------|
| Simple RNN | Basic recurrent layer, struggles with long sequences |
| Bidirectional LSTM | Gated memory, processes both directions, better for long dependencies |
| LSTM with Packed Sequences | Efficient padding handling, faster training |

### Findings:
1. **LSTM outperforms simple RNN** due to better gradient flow through gates
2. **Bidirectional processing** captures context from both directions
3. **Packed sequences** improve efficiency by skipping padding tokens
4. **Dropout** helps prevent overfitting on IMDB dataset

## Test with Custom Text

```python
def predict_sentiment(model, text, vocab, tokenizer, pad_idx):
    model.eval()
    unk_idx = vocab.stoi['<unk>']
    encoded = [vocab.stoi.get(token, unk_idx) for token in tokenizer(text)]
    tensor = torch.tensor([encoded], dtype=torch.long).to(device)
    length = torch.tensor([len(encoded)], dtype=torch.long)
    
    with torch.no_grad():
        output = model(tensor, length) if hasattr(model, 'forward') and 'lengths' in model.forward.__code__.co_varnames else model(tensor)
        prob = torch.softmax(output, dim=1)
        prediction = torch.argmax(prob, dim=1).item()
    
    return classes[prediction], prob[0][prediction].item()

# Test examples
test_reviews = [
    "This movie was absolutely fantastic! Great acting and story.",
    "Terrible film. Complete waste of time. Avoid at all costs.",
    "The plot was confusing but the visuals were stunning.",
    "An average movie, nothing special but not bad either."
]

print("Sentiment Predictions:\n")
for review in test_reviews:
    sentiment, confidence = predict_sentiment(model_packed, review, vocab, tokenizer, pad_idx)
    print(f"Review: \"{review}\"")
    print(f"Prediction: {sentiment} (confidence: {confidence:.4f})\n")
```

---

## Challenge

Using the notebooks associated to this lesson (either the PyTorch or the TensorFlow version), rerun them using your own dataset, perhaps one from Kaggle, used with attribution. Try a different kind of dataset and document your findings, using text such as [this Kaggle competition dataset about weather tweets](https://www.kaggle.com/competitions/crowdflower-weather-twitter/data?select=train.csv).
