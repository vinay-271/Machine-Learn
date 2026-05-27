# RNN Sentiment Analysis on IMDB Reviews

This repository contains a Jupyter notebook (`RNN_Sentiment.ipynb`) that implements a sentiment analysis pipeline using PyTorch. The goal is to classify movie reviews from the IMDB dataset as either positive or negative.

The model uses a Recurrent Neural Network (RNN) trained on TF-IDF features extracted from preprocessed text data.

---

## Workflow Overview

The notebook is structured into the following sections:

1. **Data Loading & Cleaning**: Reads the dataset, handles missing values, and removes duplicate entries.
2. **Text Preprocessing**: Clean the text data to prepare it for vectorization.
3. **Encoding & Vectorization**: Encodes target labels and converts textual reviews into numerical features using TF-IDF.
4. **Data Loading**: Splits the data into training and testing sets and loads them into PyTorch `DataLoaders`.
5. **Model Architecture**: Defines a basic Recurrent Neural Network (RNN) in PyTorch.
6. **Training**: Trains the network over 10 epochs.
7. **Evaluation**: Evaluates the model on the test dataset.

---

## Requirements

To run this notebook, you will need Python 3 and the following libraries:

*   `pandas`
*   `numpy`
*   `scikit-learn`
*   `nltk`
*   `torch`

You can install the required packages using pip:

```bash
pip install pandas numpy scikit-learn nltk torch
```

---

## Detailed Steps

### 1. Data Preprocessing
The text reviews undergo several preprocessing steps:
*   **Case Folding**: Converting all text to lowercase.
*   **URL Removal**: Stripping out hyperlinks using regular expressions.
*   **Punctuation Removal**: Retaining only alphanumeric characters and whitespace.
*   **HTML Tag Removal**: Removing any lingering HTML markup.
*   **Stopword Removal**: Filtering out common English stopwords using the NLTK library.
*   **Stemming**: Reducing words to their base form using NLTK's `PorterStemmer`.

### 2. Label Encoding & Vectorization
*   **Labels**: The target column (`sentiment`) is converted into binary values ($1$ for positive, $0$ for negative) using `LabelEncoder`.
*   **Features**: A `TfidfVectorizer` is configured to keep the top 5,000 most frequent words, creating a sparse matrix representation of the reviews.

### 3. PyTorch Dataset Configuration
The TF-IDF arrays are converted into PyTorch tensors.
*   The data is split into $80\%$ training and $20\%$ test sets.
*   `TensorDataset` and `DataLoader` classes are utilized to manage batching (batch size of 64) and shuffling.

### 4. Model Architecture
The network is built using PyTorch's `nn.Module`:

*   **Input Layer**: Accepts the 5,000-dimensional TF-IDF vectors.
*   **RNN Layer**: A single-layer RNN with a hidden size of 128. Note that the TF-IDF feature vector is unsqueezed to represent a sequence of length 1 before entering the RNN.
*   **Fully Connected Layer**: Maps the final hidden state to a single output value, which is then processed through a sigmoid activation function during training to obtain a probability score.

```python
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out
```

### 5. Training and Evaluation
*   **Loss Function**: Binary Cross-Entropy Loss (`BCELoss`).
*   **Optimizer**: Adam Optimizer.
*   **Epochs**: 10.
*   **Metric**: Accuracy.

---

## Results

After 10 epochs of training, the baseline model achieved an accuracy of approximately **85.66%** on the test set.
