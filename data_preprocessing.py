# Importing all necessary libraries
import nltk                           # Natural Language Toolkit – helps process text data
from nltk.stem.porter import PorterStemmer   # Used to reduce words to their root form (e.g., "running" → "run")
from nltk.corpus import stopwords     # Common words like "is", "the", "and" – usually removed
import string                         # Handles punctuation marks like .,!? etc.
import pandas as pd 

# Download required language data files (needed only once)
# nltk.download('stopwords')
# nltk.download('punkt')

# Create an object of the stemmer (used later for converting words to their root form)
ps = PorterStemmer()

# Function to clean and simplify text before giving it to a machine learning model
def transform_text(text):
    # Step 1: Convert all letters to lowercase (so "Hello" and "hello" are treated the same)
    text = text.lower()
    
    # Step 2: Split the text into individual words (called "tokens")
    text = nltk.word_tokenize(text)
    
    # Step 3: Keep only words and numbers (remove special characters like @, #, $, etc.)
    y = []
    for i in text:
        if i.isalnum():               # isalnum() means: only alphabets or numbers
            y.append(i)
            
    # Step 4: Remove common useless words (like “is”, “the”, “and”) and punctuation marks
    text = y[:]                       # Copy clean tokens to text
    y.clear()                         # Empty the list for reuse
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
        
    # Step 5: Convert each remaining word to its base form (called "stemming")
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))          # Example: "running" → "run", "studies" → "studi"
    
    # Step 6: Join the cleaned words back into one string
    return " ".join(y)

# Example: running the function on a sample sentence
# print(transform_text("hii my name is pushkar singh"))

# Output:
# "hii name pushkar singh"
#
# Explanation:
# "my" and "is" are removed (they are stopwords),
# all letters are lowercase,
# and words are stemmed (if needed).
if "__main__" == __name__:
    df = pd.read_csv("CSV_files/spam_v3.csv")

    df['transformed_text'] = df['text'].apply(transform_text)

    df.to_csv("CSV_files/spam_final.csv")