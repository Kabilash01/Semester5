import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import pandas as pd

# Download required NLTK data - FIXED
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')  # This was missing
nltk.download('omw-1.4')

class SimpleTextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
    
    def get_wordnet_pos(self, word):
        """Map POS tag to wordnet POS for better lemmatization"""
        try:
            tag = nltk.pos_tag([word])[0][1][0].upper()
            tag_dict = {
                "J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV
            }
            return tag_dict.get(tag, wordnet.NOUN)
        except:
            return wordnet.NOUN  # Default fallback
    
    def remove_punctuation(self, text):
        """Remove all punctuation marks"""
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def tokenize_text(self, text):
        """Convert text to lowercase and tokenize into words"""
        text = text.lower()
        tokens = word_tokenize(text)
        return tokens
    
    def lemmatize_tokens(self, tokens):
        """Lemmatize tokens with proper POS tagging"""
        lemmatized_tokens = []
        for token in tokens:
            if token.isalpha():  # Only process alphabetic tokens
                pos = self.get_wordnet_pos(token)
                lemmatized_token = self.lemmatizer.lemmatize(token, pos)
                lemmatized_tokens.append(lemmatized_token)
        return lemmatized_tokens
    
    def preprocess_text(self, text):
        """Complete preprocessing pipeline"""
        if not isinstance(text, str):
            return ""
        
        # Step 1: Remove punctuation
        text_no_punct = self.remove_punctuation(text)
        
        # Step 2: Tokenization (convert to lowercase and split into words)
        tokens = self.tokenize_text(text_no_punct)
        
        # Step 3: Lemmatization
        lemmatized_tokens = self.lemmatize_tokens(tokens)
        
        # Join tokens back into text
        processed_text = ' '.join(lemmatized_tokens)
        
        return processed_text
    
    def preprocess_dataframe(self, df, text_column):
        """Preprocess text column in pandas DataFrame"""
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        processed_texts = []
        for text in df[text_column]:
            processed_text = self.preprocess_text(text)
            processed_texts.append(processed_text)
        
        return processed_texts

# Alternative simple approach without POS tagging (faster and more reliable)
class BasicTextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
    
    def preprocess_text(self, text):
        """Basic preprocessing: punctuation removal, tokenization, lemmatization"""
        if not isinstance(text, str):
            return ""
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Convert to lowercase and tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove non-alphabetic tokens and lemmatize
        lemmatized_tokens = []
        for token in tokens:
            if token.isalpha():
                lemmatized_token = self.lemmatizer.lemmatize(token)
                lemmatized_tokens.append(lemmatized_token)
        
        return ' '.join(lemmatized_tokens)
    
    def preprocess_dataframe(self, df, text_column):
        """Preprocess text column in pandas DataFrame"""
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        processed_texts = []
        for text in df[text_column]:
            processed_text = self.preprocess_text(text)
            processed_texts.append(processed_text)
        
        return processed_texts

# Function to preprocess a list of texts
def preprocess_reviews(texts, use_pos_tagging=False):  # Changed default to False
    """
    Preprocess a list of review texts
    
    Args:
        texts: List of strings (reviews)
        use_pos_tagging: Boolean, whether to use POS tagging for better lemmatization
    
    Returns:
        List of preprocessed texts
    """
    if use_pos_tagging:
        preprocessor = SimpleTextPreprocessor()
    else:
        preprocessor = BasicTextPreprocessor()
    
    processed_texts = []
    for text in texts:
        processed_text = preprocessor.preprocess_text(text)
        processed_texts.append(processed_text)
    
    return processed_texts

# Example usage for batch processing
def batch_preprocess_example():
    """Example of preprocessing multiple reviews"""
    reviews = [
        "The camera quality was amazing ! but battery life was not impressive and charging takes forever",
        "I love this product!!! It's working perfectly and delivery was fast.",
        "Not good at all. The quality is terrible and very expensive.",
        "Excellent build quality, amazing features, but price is too high!",
        "Average product, nothing special about it."
    ]
    
    print("Original reviews:")
    for i, review in enumerate(reviews, 1):
        print(f"{i}. {review}")
    
    print("\n" + "="*60 + "\n")
    
    # Preprocess all reviews using basic approach (more reliable)
    processed_reviews = preprocess_reviews(reviews, use_pos_tagging=False)
    
    print("Processed reviews:")
    for i, processed_review in enumerate(processed_reviews, 1):
        print(f"{i}. {processed_review}")

def main():
    """Main function to demonstrate all functionality"""
    # Sample product review
    sample_review = "The camera quality was amazing ! but battery life was not impressive and charging takes forever"
    
    print("Original text:")
    print(sample_review)
    print("\n" + "="*60 + "\n")
    
    # Method 2: Basic approach (faster and more reliable)
    preprocessor2 = BasicTextPreprocessor()
    processed_text2 = preprocessor2.preprocess_text(sample_review)
    
    print("Processed text (basic approach):")
    print(processed_text2)
    print("\n" + "="*60 + "\n")
    
    # Try POS tagging approach (might need additional downloads)
    try:
        preprocessor1 = SimpleTextPreprocessor()
        processed_text1 = preprocessor1.preprocess_text(sample_review)
        print("Processed text (with POS tagging):")
        print(processed_text1)
        print("\n" + "="*60 + "\n")
    except Exception as e:
        print(f"POS tagging failed: {e}")
        print("Using basic approach instead...\n" + "="*60 + "\n")
    
    # Example with DataFrame
    sample_data = {
        'review_id': [1, 2, 3],
        'review_text': [
            "The camera quality was amazing ! but battery life was not impressive and charging takes forever",
            "I love this product!!! It's working perfectly and delivery was fast.",
            "Not good at all. The quality is terrible and very expensive."
        ],
        'rating': [3, 5, 1]
    }
    
    df = pd.DataFrame(sample_data)
    
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*60 + "\n")
    
    # Preprocess the review_text column using reliable basic approach
    df['processed_text'] = preprocessor2.preprocess_dataframe(df, 'review_text')
    
    print("DataFrame with processed text:")
    print(df[['review_text', 'processed_text']])
    print("\n" + "="*60 + "\n")
    
    # Run batch processing example
    print("BATCH PROCESSING EXAMPLE:")
    print("="*60)
    batch_preprocess_example()

# Example usage
if __name__ == "__main__":
    main()