#!/usr/bin/env python3
"""
Simple test script to verify that the source modules can be imported
and basic functionality works.
"""

import sys
import os

# Add source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'source'))

print("Testing source modules...")
print("-" * 60)

# Test imports
try:
    print("✓ Testing image_processing module...")
    from source import image_processing
    print("  - Image processing module loaded successfully")
    
    print("✓ Testing text_processing module...")
    from source import text_processing
    print("  - Text processing module loaded successfully")
    
    print("✓ Testing cnn_models module...")
    from source import cnn_models
    print("  - CNN models module loaded successfully")
    
    print("✓ Testing training module...")
    from source import training
    print("  - Training module loaded successfully")
    
    print("✓ Testing inference module...")
    from source import inference
    print("  - Inference module loaded successfully")
    
    print("-" * 60)
    
    # Test some basic functionality
    print("\nTesting basic functionality...")
    print("-" * 60)
    
    # Test text processing
    sample_text = "This is a SAMPLE Medical Text with    extra spaces."
    cleaned = text_processing.clean_text(sample_text)
    print(f"✓ Text cleaning: '{sample_text}' -> '{cleaned}'")
    
    # Test text statistics
    stats = text_processing.calculate_text_statistics(sample_text)
    print(f"✓ Text statistics: {stats['word_count']} words, {stats['character_count']} characters")
    
    # Test model creation
    model = cnn_models.get_model('simple', num_classes=2, input_channels=3)
    print(f"✓ Model creation: {model.__class__.__name__} created successfully")
    
    print("-" * 60)
    print("\n✅ All tests passed! Source modules are working correctly.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
