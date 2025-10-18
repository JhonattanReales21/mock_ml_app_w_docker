"""
Text processing utilities for medical text analysis and NLP tasks.
"""

import re
import string
from typing import List, Dict


def clean_text(text):
    """
    Clean and normalize medical text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def remove_punctuation(text):
    """
    Remove punctuation from text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text without punctuation
    """
    return text.translate(str.maketrans('', '', string.punctuation))


def tokenize(text):
    """
    Simple tokenization by splitting on whitespace.
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of tokens
    """
    return text.split()


def extract_medical_terms(text, medical_terms_list=None):
    """
    Extract medical terms from text.
    
    Args:
        text (str): Input text
        medical_terms_list (list): List of medical terms to search for
        
    Returns:
        list: Found medical terms
    """
    if medical_terms_list is None:
        # Default list of common medical terms
        medical_terms_list = [
            'diagnosis', 'treatment', 'patient', 'symptoms', 'disease',
            'medication', 'surgery', 'therapy', 'condition', 'infection',
            'cancer', 'diabetes', 'hypertension', 'cardiac', 'respiratory'
        ]
    
    text_lower = text.lower()
    found_terms = []
    
    for term in medical_terms_list:
        if term.lower() in text_lower:
            found_terms.append(term)
    
    return found_terms


def anonymize_patient_info(text):
    """
    Anonymize patient identifiable information in medical text.
    
    Args:
        text (str): Input medical text
        
    Returns:
        str: Anonymized text
    """
    # Replace potential phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
    
    # Replace potential email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    
    # Replace potential dates (various formats)
    text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[DATE]', text)
    
    # Replace potential ID numbers
    text = re.sub(r'\b[A-Z]{2}\d{6,}\b', '[ID]', text)
    
    return text


def extract_keywords(text, top_n=10):
    """
    Extract keywords from medical text (simple frequency-based).
    
    Args:
        text (str): Input text
        top_n (int): Number of top keywords to return
        
    Returns:
        list: List of tuples (keyword, frequency)
    """
    # Clean and tokenize
    cleaned = clean_text(text)
    cleaned = remove_punctuation(cleaned)
    tokens = tokenize(cleaned)
    
    # Common stopwords to exclude
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was',
                 'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do',
                 'does', 'did', 'will', 'would', 'could', 'should', 'may',
                 'might', 'can', 'that', 'this', 'these', 'those'}
    
    # Filter stopwords and count frequencies
    word_freq = {}
    for token in tokens:
        if token not in stopwords and len(token) > 2:
            word_freq[token] = word_freq.get(token, 0) + 1
    
    # Sort by frequency and return top N
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return sorted_words[:top_n]


def create_text_summary(text, max_sentences=3):
    """
    Create a simple summary of the text by extracting first N sentences.
    
    Args:
        text (str): Input text
        max_sentences (int): Maximum number of sentences in summary
        
    Returns:
        str: Summary text
    """
    # Split into sentences (simple approach)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Return first N sentences
    summary_sentences = sentences[:max_sentences]
    return '. '.join(summary_sentences) + '.'


def calculate_text_statistics(text):
    """
    Calculate basic statistics about the text.
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Dictionary with text statistics
    """
    tokens = tokenize(text)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return {
        'character_count': len(text),
        'word_count': len(tokens),
        'sentence_count': len(sentences),
        'avg_word_length': sum(len(word) for word in tokens) / len(tokens) if tokens else 0,
        'avg_sentence_length': len(tokens) / len(sentences) if sentences else 0
    }
