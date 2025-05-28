# SMS Spam Classification System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

A machine learning-powered SMS spam detection system built with Python, featuring an interactive web interface for real-time message classification. The system uses TF-IDF vectorization and logistic regression to achieve high accuracy in distinguishing between spam and legitimate (ham) messages.

## Features

- **High-Performance Classification**: Achieves 97%+ accuracy using optimized TF-IDF features
- **Interactive Testing Interface**: Real-time message classification with adjustable threshold
- **Dynamic Preprocessing**: Configurable n-gram ranges, stop words, and feature limits
- **Threshold Optimization**: Automated F1-score based threshold tuning
- **Feature Analysis**: Visualization of most important spam/ham indicators
- **Export Capabilities**: CSV export of prediction logs for analysis
- **Model Persistence**: Save and load trained models for production use

## Quick Start

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib ipywidgets
```

### Installation

Clone the repository:
```bash
git clone https://github.com/vijaybartaula/sms-spam-classifier.git
cd sms-spam-classifier
```

### Usage

The system automatically:
1. Downloads the SMS dataset
2. Performs exploratory data analysis
3. Trains the logistic regression model
4. Optimizes the classification threshold
5. Launches an interactive testing interface

## Dataset

- **Source**: SMS Spam Collection Dataset
- **Size**: 5,574 messages
- **Format**: Tab-separated values (label, message)
- **Classes**: Ham (legitimate) and Spam messages
- **Distribution**: ~87% Ham, ~13% Spam

## Model Architecture

### Text Preprocessing
- **Tokenization**: Standard whitespace tokenization
- **Case Normalization**: Lowercase conversion
- **Stop Words**: English + custom SMS-specific stop words
- **N-grams**: Configurable unigrams and bigrams

### Feature Engineering
- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Feature Selection**: Top 10,000 most informative features
- **Dimensionality**: Sparse matrix representation for efficiency

### Classification
- **Algorithm**: Logistic Regression with L2 regularization
- **Threshold**: Optimized using F1-score maximization
- **Training**: 80% train, 20% test split with stratification

## Configuration Parameters

Modify these variables in the code to customize the model:

```python
# Preprocessing Parameters
custom_stopwords = {'u', 'ur', '4', '2', 'im', 'dont', 'd'}
use_custom_stopwords = True

# Vectorization Parameters
ngram_min = 1
ngram_max = 2
max_features = 10000

# Classification Parameters
threshold_default = 0.5
```

## Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 97.3% |
| Precision (Spam) | 95.8% |
| Recall (Spam) | 91.2% |
| F1-Score | 93.4% |
| False Positive Rate | <2% |

## Interactive Features

### Real-time Classification
- Input any message for instant spam/ham classification
- View probability scores for both classes
- Adjust classification threshold dynamically

### Prediction Logging
- Automatic logging of all predictions
- Export logs to CSV for further analysis
- View recent prediction history

### Visualization
- Class distribution analysis
- Message length distribution comparison
- Feature importance visualization
- Confusion matrix heatmap

## API Reference

### Core Functions

#### `find_best_threshold(probs, true_labels)`
Finds optimal classification threshold using F1-score maximization.

**Parameters:**
- `probs`: Array of spam probabilities
- `true_labels`: Ground truth labels

**Returns:**
- `best_threshold`: Optimal threshold value
- `best_f1`: Maximum F1-score achieved

### Model Loading

```python
import joblib

# Load pre-trained model and vectorizer
model = joblib.load('spam_classifier_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Classify new message
def classify_message(message, threshold=0.5):
    msg_vec = vectorizer.transform([message])
    spam_prob = model.predict_proba(msg_vec)[0][1]
    return 'Spam' if spam_prob >= threshold else 'Ham'
```

## Advanced Configuration

### Custom Stop Words
Add domain-specific terms to improve classification:

```python
custom_stopwords = {
    'u', 'ur', '4', '2', 'im', 'dont', 'd',  # SMS abbreviations
    'txt', 'msg', 'call', 'free', 'win'      # Common spam terms
}
```

### Feature Engineering
Experiment with different vectorization parameters:

```python
vectorizer = TfidfVectorizer(
    stop_words=stop_words,
    lowercase=True,
    ngram_range=(1, 3),        # Include trigrams
    max_features=15000,        # Increase vocabulary
    min_df=2,                  # Minimum document frequency
    max_df=0.95               # Maximum document frequency
)
```

## Production Deployment

### Model Serving
For production use, implement a REST API:

```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('spam_classifier_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

@app.route('/classify', methods=['POST'])
def classify():
    message = request.json['message']
    msg_vec = vectorizer.transform([message])
    spam_prob = model.predict_proba(msg_vec)[0][1]
    return jsonify({
        'message': message,
        'spam_probability': float(spam_prob),
        'classification': 'spam' if spam_prob >= 0.5 else 'ham'
    })
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- SMS Spam Collection Dataset by Tiago A. Almeida and José María Gómez Hidalgo
- scikit-learn community for machine learning tools
- Jupyter/IPython for interactive computing capabilities

## Support

For questions, bug reports, or feature requests:
- Open an issue on GitHub
- Documentation: See `whitepaper.md` for technical details

## Changelog

### v1.0.0 (Current)
- Initial release with core classification functionality
- Interactive testing interface
- Model persistence and threshold optimization
- Comprehensive visualization suite
