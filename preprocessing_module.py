"""
Text Preprocessing Module for Sentiment Analysis
Handles all text cleaning and preprocessing operations
"""

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """
    A comprehensive text preprocessing class for sentiment analysis
    """
    
    def __init__(self, download_nltk=True):
        """
        Initialize the preprocessor
        
        Args:
            download_nltk (bool): Whether to download required NLTK data
        """
        self.lemmatizer = None
        self.stop_words = set()
        
        if download_nltk:
            self._download_nltk_requirements()
        
        self._initialize_tools()
    
    def _download_nltk_requirements(self):
        """Download required NLTK data"""
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            nltk.download('punkt', quiet=True)
            logger.info("NLTK data downloaded successfully")
        except Exception as e:
            logger.warning(f"Could not download NLTK data: {e}")
    
    def _initialize_tools(self):
        """Initialize preprocessing tools"""
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            logger.info("Preprocessing tools initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize preprocessing tools: {e}")
            self.lemmatizer = None
            self.stop_words = set()
    
    def clean_text(self, text):
        """
        Clean and preprocess a single text
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        # Handle missing/null values
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to string (safety check)
        text = str(text)
        
        # Remove URLs, email addresses, and mentions
        text = re.sub(r'http\S+|www\S+|@\S+', '', text)
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove non-alphabetical characters
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize by splitting on whitespace
        words = text.split()
        
        # Remove stopwords, lemmatize, and filter short words
        if self.lemmatizer and self.stop_words:
            words = [self.lemmatizer.lemmatize(word) for word in words 
                     if word not in self.stop_words and len(word) > 2]
        else:
            # Fallback without lemmatization
            words = [word for word in words if len(word) > 2]
        
        return ' '.join(words)
    
    def preprocess_batch(self, texts):
        """
        Preprocess a batch of texts
        
        Args:
            texts (list): List of texts to preprocess
            
        Returns:
            list: List of cleaned texts
        """
        return [self.clean_text(text) for text in texts]
    
    def preprocess_dataframe(self, df, text_column, output_column=None):
        """
        Preprocess texts in a pandas DataFrame
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of column containing text
            output_column (str): Name of output column (default: text_column + '_clean')
            
        Returns:
            pd.DataFrame: DataFrame with cleaned text column
        """
        if output_column is None:
            output_column = f"{text_column}_clean"
        
        # Ensure text column is string type
        df[text_column] = df[text_column].astype(str)
        
        # Apply preprocessing
        df[output_column] = df[text_column].apply(self.clean_text)
        
        logger.info(f"Preprocessed {len(df)} texts in column '{text_column}'")
        
        return df

def rating_to_sentiment(rating):
    """
    Convert numerical rating to sentiment label
    
    Args:
        rating (int/float): Numerical rating
        
    Returns:
        str: Sentiment label (Positive, Negative, Neutral, Unknown)
    """
    if pd.isna(rating):
        return "Unknown"
    
    try:
        rating = float(rating)
        if rating in [1, 2]:
            return "Negative"
        elif rating == 3:
            return "Neutral"
        elif rating in [4, 5]:
            return "Positive"
        else:
            return "Unknown"
    except (ValueError, TypeError):
        return "Unknown"

def get_text_statistics(text):
    """
    Get basic statistics about a text
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Dictionary with text statistics
    """
    if pd.isna(text) or text == '':
        return {
            'char_count': 0,
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0
        }
    
    text = str(text)
    words = text.split()
    sentences = text.count('.') + text.count('!') + text.count('?')
    
    return {
        'char_count': len(text),
        'word_count': len(words),
        'sentence_count': max(1, sentences),  # At least 1 sentence
        'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0
    }

# Example usage and testing
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Test texts
    test_texts = [
        "This product is absolutely AMAZING!!! I love it so much 😍",
        "This is the worst service I've ever experienced. Terrible!",
        "It's okay, nothing special but not bad either.",
        "Check out this link: https://example.com @user #hashtag"
    ]
    
    print("Testing Text Preprocessing:")
    print("=" * 50)
    
    for i, text in enumerate(test_texts, 1):
        cleaned = preprocessor.clean_text(text)
        stats = get_text_statistics(text)
        
        print(f"\nTest {i}:")
        print(f"Original:  {text}")
        print(f"Cleaned:   {cleaned}")
        print(f"Stats:     {stats}")
    
    # Test rating conversion
    print("\nTesting Rating to Sentiment Conversion:")
    print("=" * 50)
    
    test_ratings = [1, 2, 3, 4, 5, 2.5, None, "invalid"]
    for rating in test_ratings:
        sentiment = rating_to_sentiment(rating)
        print(f"Rating {rating} -> {sentiment}")
