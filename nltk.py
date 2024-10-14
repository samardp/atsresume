import nltk
import os

# Set up a path within the home directory for nltk data
home_dir = os.path.expanduser("~")
nltk_data_dir = os.path.join(home_dir, 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

# Ensure punkt is downloaded
nltk.download('punkt', download_dir=nltk_data_dir)
