import nltk
import os

# Define the NLTK data directory
nltk_data_dir = '/path/to/nltk_data'  # Change this to the path where nltk_data should be stored
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

# Ensure punkt is downloaded
nltk.download('punkt', download_dir=nltk_data_dir)

