# FPT Play Sentiment Analysis Project

A comprehensive sentiment analysis project analyzing user comments on the FPT Play platform using Vietnamese natural language processing models.

##  Project Overview

This project performs sentiment analysis on 50,000 user comments from the FPT Play streaming platform collected in 2015. Using a pre-trained Vietnamese sentiment analysis model based on PhoBERT, we classify user comments into positive, negative, or neutral sentiments to understand user emotions and feedback.

##  Key Findings

- **Dataset**: 50,000 comments from 26,283 users across 3,421 pieces of content
- **Sentiment Distribution**: 
  - Negative: 38.35%
  - Neutral: 32.32%
  - Positive: 29.32%
- **Peak Activity**: August 2015 (~8,000 comments) coinciding with SEA Games 28
- **Top Categories**: Events (live broadcasts) and VOD (Video on Demand)

##  Technology Stack

- **Python 3.11+**
- **PyTorch** - Deep learning framework
- **Transformers** (Hugging Face) - Pre-trained model integration
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **pymongo** - MongoDB database connection
- **Jupyter Notebook** - Development environment

## Model Information

This project uses the **PhoBERT-based Vietnamese Sentiment Analysis** model:
- **Base Model**: PhoBERT (Vietnamese BERT by VinAI)
- **Fine-tuned Model**: `wonrax/phobert-base-vietnamese-sentiment`
- **Training Data**: 30,000 Vietnamese e-commerce comments
- **Output Classes**: 
  - 0: Negative
  - 1: Positive  
  - 2: Neutral


## Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/sentiment-analysis-fpt-play.git
cd sentiment-analysis-fpt-play
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

### Running the Analysis

1. **Open the Jupyter notebook**
```bash
jupyter notebook "Sentiment Analysis - TranXuanTien.ipynb"
```

2. **Data Retrieval**
```python
# Connect to MongoDB and retrieve comments
df = retrieve_comments_data(Nrecords=50000)
```

3. **Text Preprocessing**
```python
# Standardize Vietnamese text
df['content_std'] = df['content'].apply(standardize_data)
```

4. **Sentiment Analysis**
```python
# Apply sentiment analysis model
df['sentiment'] = df['content_std'].apply(sentiment)
```

### Key Functions

**Data Preprocessing**
```python
def standardize_data(text):
    """
    Standardize Vietnamese text by:
    - Converting to lowercase
    - Removing punctuation
    - Standardizing abbreviations (ko -> không, đc -> được)
    - Cleaning whitespace
    """
```

**Sentiment Prediction**
```python
def sentiment(sentence):
    """
    Predict sentiment using PhoBERT model
    Returns: 0 (negative), 1 (positive), 2 (neutral)
    """
```

## 📈 Key Insights

### Content Analysis
- **Most Commented**: Live sports events (SEA Games 28)
- **Device Usage**: iOS (21K), Android (12K), Web (12K)
- **High Negative Sentiment**: Sports and competitive content (up to 60%)

### Business Recommendations

1. **Service Quality Focus**: Address technical issues like lagging, video quality
2. **Content Strategy**: Understand that competitive content naturally generates mixed emotions
3. **User Experience**: Improve commentator knowledge and streaming quality for live events

## Data Processing Pipeline

1. **Data Extraction**: MongoDB → Pandas DataFrame
2. **Text Cleaning**: Remove punctuation, standardize abbreviations
3. **Model Application**: PhoBERT sentiment classification
4. **Analysis**: Statistical analysis and visualization
5. **Export**: Results saved to CSV format

## Performance Metrics

- **Processing Time**: ~45 seconds for 1,000 comments
- **Model Accuracy**: Based on PhoBERT fine-tuned performance
- **Categories Analyzed**: Events, VOD, Highlights, Channels

## Acknowledgments

- **VinAI Research** for the PhoBERT model
- **Pham Huu Quang** for the Vietnamese sentiment analysis model
- **FPT Play** for the dataset
- **Hugging Face** for the Transformers library

## References

- [PhoBERT: Pre-trained language models for Vietnamese](https://github.com/VinAIResearch/PhoBERT)
- [Vietnamese Sentiment Analysis Model](https://huggingface.co/wonrax/phobert-base-vietnamese-sentiment)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)

---

**Note**: This project is for educational and research purposes. Please ensure compliance with data privacy regulations when working with user-generated content.
