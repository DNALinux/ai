import os
import re
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text


def preprocess_files(input_directory, output_directory):
    input_dir = Path(input_directory)
    output_dir = Path(output_directory)
    
    # Create output directory if it doesn't exist
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    # Process each file in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.txt'):
            with open(input_dir / file_name, 'r', encoding='utf-8') as file:
                text = file.read()
                preprocessed_text = preprocess_text(text)
            
            # Save preprocessed text to output directory
            output_file = output_dir / file_name
            with open(output_file, 'w', encoding='utf-8') as file:
                file.write(preprocessed_text)
            
            print(f"Preprocessed {file_name} and saved to {output_file}")