import re
import joblib
import argparse
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to preprocess the entered text
def preprocess_text(text):
    # Remove unwanted characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    # Tokenize
    tokens = word_tokenize(text)
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Join tokens into a cleaned string
    cleaned_text = ' '.join(lemmatized_tokens)
    return cleaned_text

def process_text(text, vect_path):
    preprocessed_documents = [preprocess_text(doc) for doc in text]
    vectorizer = joblib.load(vect_path)
    vectors = vectorizer.transform(preprocessed_documents)
    return vectors

# Function to load and classify using a given model
def classify_text(text, model_path, vect_path):
    # Preprocess the text
    preprocessed_text = process_text(text, vect_path)

    # Load the model
    model = joblib.load(model_path)

    # Classify the text
    category = model.predict(preprocessed_text)
    
    return category
def get_category(num):
    category_dict = {0: "alt.atheism", 1: "comp.graphics", 2: "comp.os.ms-windows.misc", 3: "comp.sys.ibm.pc.hardware", 4: "comp.sys.mac.hardware", 5: "comp.windows.x", 6: "misc.forsale", 7: "rec.autos", 8: "rec.motorcycles", 9: "rec.sport.baseball", 10: "rec.sport.hockey", 11: "sci.crypt", 12: "sci.electronics", 13: "sci.med", 14: "sci.space", 15: "soc.religion.christian", 16: "talk.politics.guns", 17: "talk.politics.mideast", 18: "talk.politics.misc", 19: "talk.religion.misc"}
    return category_dict[num]

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Classify text into 20 Newsgroups categories.")
    parser.add_argument("text", help="The text to classify.")
    parser.add_argument("--model", choices=["knn", "nb", "dt", "svm"], default="knn", help="Choose the classification model (knn, nb, dt). Default is knn.")

    args = parser.parse_args()

    # Choose the model path based on the user's choice
    model_path_dict = {"knn": "model/knn/knn_model.joblib", "nb": "model/nb/naive_bayes_model.joblib", "dt": "model/dt/decision_tree_model.joblib","svm":"model/svm/svm_model.joblib",}
    vect_path_dict = {"knn": "model/knn/tfidf_vectorizer.joblib", "nb": "model/nb/tfidf_vectorizer.joblib", "dt": "model/dt/tfidf_vectorizer.joblib","svm":"model/svm/tfidf_vectorizer.joblib"}
    model_path = model_path_dict[args.model]
    vect_path = vect_path_dict[args.model]
    print(model_path)
    # Classify the text
#     text = """RSA is a crypto system which is asymmetric, or public-key.  This means
#  that there are two different, related keys: one to encrypt and one to
#  decrypt.  Because one cannot (reasonably) be derived from the other,
#  you may publish your encryption, or public key widely and keep your
#  decryption, or private key to yourself.  Anyone can use your public
#  key to encrypt a message, but only you hold the private key needed to
#  decrypt it.  (Note that the "message" sent with RSA is normally just
#  the DES key to the real message."""
    category = classify_text([args.text], model_path, vect_path)
    category = get_category(category[0])
    print(f"The text belongs to 20 Newsgroups category: {category}")
