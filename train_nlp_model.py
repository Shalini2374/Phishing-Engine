import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import string # To remove punctuation

# Ensure NLTK data is available (run this once interactively if needed)
# nltk.download('punkt')
# nltk.download('stopwords')

# --- Text Preprocessing Function ---
def preprocess_text(text):
    """
    Cleans and prepares text data for NLP:
    1. Lowercase
    2. Remove punctuation
    3. Tokenize (split into words)
    4. Remove stopwords
    """
    if not isinstance(text, str):
        return "" # Return empty string for non-string inputs (like NaN)

    # 1. Lowercase
    text = text.lower()
    # 2. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 3. Tokenize
    tokens = word_tokenize(text)
    # 4. Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words] # Keep alphanumeric words
    
    return " ".join(filtered_tokens) # Rejoin tokens into a string

# --- Main Training Script ---
def main():
    print("--- Phase 3: Training NLP Content Analyzer Model ---")

    # 1. Load the dataset
    print("Loading dataset...")
    try:
        df = pd.read_csv("master_dataset.csv")
        # Ensure we have text and labels
        df = df.dropna(subset=['page_text', 'label'])
        print(f"Loaded {len(df)} rows with text content.")
    except FileNotFoundError:
        print("[ERROR] master_dataset.csv not found! Did Phase 1 complete successfully?")
        return
    except Exception as e:
        print(f"[ERROR] Could not load dataset: {e}")
        return

    # 2. Preprocess the text data
    print("Preprocessing text data (this may take a few minutes)...")
    # Apply the cleaning function to the 'page_text' column
    df['cleaned_text'] = df['page_text'].apply(preprocess_text)
    print("Text preprocessing complete.")
    print("Sample cleaned text:")
    print(df[['page_text', 'cleaned_text', 'label']].head())

    # Prepare data for model
    X_text = df['cleaned_text']
    y = df['label']

    # 3. Split data into training and testing sets
    print("\nSplitting data into train/test sets (80/20)...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set size: {len(X_train_text)}")
    print(f"Testing set size: {len(X_test_text)}")

    # 4. Vectorize text using TF-IDF
    print("\nVectorizing text using TF-IDF...")
    # max_features limits the vocabulary size, common practice for text
    # ngram_range=(1, 2) considers both single words and pairs of words
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text) # Use transform only on test data
    
    print(f"Created TF-IDF matrix with shape: {X_train_tfidf.shape}")

    # 5. Train a Naive Bayes model (good baseline for text classification)
    print("\nTraining Multinomial Naive Bayes model...")
    # Alpha is a smoothing parameter
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_tfidf, y_train)
    print("Model training complete.")

    # 6. Evaluate the model
    print("\nEvaluating model performance...")
    y_pred = model.predict(X_test_tfidf)

    print("\nClassification Report (NLP Model):")
    print(classification_report(y_test, y_pred, target_names=['Legitimate (0)', 'Phishing (1)']))

    print("\nConfusion Matrix (NLP Model):")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print(f"\nOverall Accuracy (NLP Model): {accuracy_score(y_test, y_pred):.4f}")

    # Optional: Show Top TF-IDF Features for Phishing (Helps understand the model)
    try:
        print("\nTop 20 TF-IDF features for Phishing class:")
        feature_names = vectorizer.get_feature_names_out()
        # Get log probabilities for each feature per class
        log_prob_phish = model.feature_log_prob_[1] # Index 1 for phishing class
        # Sort features by probability (higher is more indicative)
        sorted_features_indices = log_prob_phish.argsort()[::-1] # Descending order
        top_features = [feature_names[i] for i in sorted_features_indices[:20]]
        print(top_features)
    except Exception as e:
        print(f"Could not extract top features: {e}")


    # 7. Save the trained model AND the vectorizer
    model_filename = "nlp_model.joblib"
    vectorizer_filename = "tfidf_vectorizer.joblib"
    print(f"\nSaving trained model to {model_filename}...")
    joblib.dump(model, model_filename)
    print(f"Saving TF-IDF vectorizer to {vectorizer_filename}...")
    joblib.dump(vectorizer, vectorizer_filename) # Crucial: Need this to process new text later!
    print("Model and vectorizer saved successfully.")

    print("\n--- Phase 3 Finished ---")

if __name__ == "__main__":
    main()