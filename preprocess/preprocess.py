# Import the libraries
import re
from sklearn.datasets import fetch_20newsgroups
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Fetch the data
newsgroups = fetch_20newsgroups()

# Remove the headers manually
# This is a simple regex to match the header format
header_pattern = re.compile(r"^(From|Subject|Organization|Lines|Nntp-Posting-Host|Summary|Keywords|Reply-To|Distribution|X-Newsreader|Article-I.D.):.*\n")

# This is a list to store the cleaned documents
cleaned_data = []

# Loop through the documents
for doc in newsgroups.data:
  # Replace the headers with an empty string
  cleaned_doc = header_pattern.sub("", doc)
  # Append the cleaned document to the list
  cleaned_data.append(cleaned_doc)

# Tokenize the text
tokens = [word_tokenize(doc) for doc in cleaned_data]

# Normalize the text
lemmatizer = WordNetLemmatizer()
normalized = [[lemmatizer.lemmatize(token.lower()) for token in doc] for doc in tokens]

# Vectorize the text
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(cleaned_data)
