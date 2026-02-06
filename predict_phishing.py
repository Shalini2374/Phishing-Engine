import joblib
import pandas as pd
import numpy as np
from urllib.parse import urlparse, parse_qs
import ipaddress
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess # Use MobileNetV2 preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import os
import random
import time
from collections import Counter
import math # Needed for entropy

# --- Load Models and Vectorizer ---
try:
    print("Loading models and vectorizer...")
    # --- LOAD THE NEW XGBOOST MODEL ---
    url_model = joblib.load("url_model_xgb_no_age.joblib")
    # ---------------------------------
    nlp_model = joblib.load("nlp_model.joblib")
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.joblib")
    # Load the Keras model (CV model)
    cv_model = load_model("cv_phishing_model.keras")
    print("Models and vectorizer loaded successfully.")
except FileNotFoundError as e:
    print(f"[ERROR] Model file not found: {e}. Ensure all model files are in the PhishingEngine directory.")
    exit()
except Exception as e:
    print(f"[ERROR] Failed to load models: {e}")
    exit()


# --- Feature Extraction Functions (Copied from enhanced train_url_model.py, NO domain age) ---

# Helper: Safely parse URL
def safe_urlparse(url):
    try:
        if not url.startswith(('http://', 'https://')): url = 'http://' + url
        return urlparse(url)
    except Exception: return urlparse('')

# Helper: Calculate Shannon Entropy
def calculate_entropy(text):
    if not text: return 0.0
    text = text.lower(); counts = Counter(text); entropy = 0.0
    text_len = float(len(text))
    if text_len == 0: return 0.0 # Avoid division by zero
    for count in counts.values(): p = count / text_len; entropy -= p * math.log2(p)
    return entropy

# --- Feature Categories ---

# 1. Address Bar Features
def having_ip_address(url):
    try: netloc = safe_urlparse(url).netloc; ipaddress.ip_address(netloc); return 1
    except: return 0
def url_length_category(url):
    length = len(url)
    if length < 54: return 0
    elif 54 <= length <= 75: return 1
    else: return 2
def uses_https(url):
    try: return 1 if safe_urlparse(url).scheme == 'https' else 0
    except: return 0

# 2. Character Counts
def having_at_symbol(url): return 1 if "@" in url else 0
def count_dots(url): return url.count('.')
def count_hyphens(url): return url.count('-')
def count_slashes(url): return url.count('/')
def count_questionmarks(url): return url.count('?')
def count_equals(url): return url.count('=')
def count_percent(url): return url.count('%')
def count_underscore(url): return url.count('_')
def digit_count(url): return sum(c.isdigit() for c in url)
def letter_count(url): return sum(c.isalpha() for c in url)

# 3. Domain/Hostname Features
def domain_length(url):
    try: return len(safe_urlparse(url).netloc)
    except: return 0
def prefix_suffix(url):
    try:
        domain = safe_urlparse(url).netloc; parts = domain.split('.')
        if not domain: return 0
        if len(parts) >= 2 and ('-' in parts[-2]): return 1
        if domain.startswith('-') or domain.endswith('-'): return 1
        return 0
    except: return 0
def count_subdomains(url):
    try:
        hostname = safe_urlparse(url).hostname
        if hostname:
            parts = hostname.split('.')
            if len(parts) > 2:
                is_sld = parts[-2] in ['co', 'com', 'org', 'net', 'gov', 'edu', 'ac'] and len(parts[-1]) <= 3
                if parts[0] == 'www': base_parts = 3 if is_sld and len(parts) > 3 else 2; return max(0, len(parts) - base_parts)
                else: base_parts = 3 if is_sld and len(parts) > 3 else 2; return max(0, len(parts) - base_parts)
            return 0
        return 0
    except: return 0
def https_token_in_domain(url):
    try: domain = safe_urlparse(url).netloc; return 1 if domain and 'https' in domain.lower() else 0
    except: return 0
def domain_entropy(url):
    try: domain = safe_urlparse(url).netloc; return calculate_entropy(domain)
    except: return 0.0
def domain_digit_ratio(url):
    try:
        domain = safe_urlparse(url).netloc; length = len(domain)
        if not domain or length == 0: return 0.0
        digits = sum(c.isdigit() for c in domain); return float(digits) / length
    except: return 0.0
def domain_letter_ratio(url):
    try:
        domain = safe_urlparse(url).netloc; length = len(domain)
        if not domain or length == 0: return 0.0
        letters = sum(c.isalpha() for c in domain); return float(letters) / length
    except: return 0.0

# 4. Path Features
def path_length(url):
    try: path = safe_urlparse(url).path; return len(path) if path else 0
    except: return 0
def double_slash_redirecting(url): # Check // in path/query only
    try:
        parsed = safe_urlparse(url); path_query = parsed.path + ('?' + parsed.query if parsed.query else '')
        # Ensure path_query is a string before calling find
        if not isinstance(path_query, str): return 0
        return 1 if path_query.find('//') != -1 else 0
    except: return 0
def path_entropy(url):
    try: path = safe_urlparse(url).path; return calculate_entropy(path)
    except: return 0.0
def path_digit_ratio(url):
    try:
        path = safe_urlparse(url).path; length = len(path)
        if not path or length == 0: return 0.0
        digits = sum(c.isdigit() for c in path); return float(digits) / length
    except: return 0.0
def path_letter_ratio(url):
    try:
        path = safe_urlparse(url).path; length = len(path)
        if not path or length == 0: return 0.0
        letters = sum(c.isalpha() for c in path); return float(letters) / length
    except: return 0.0

# 5. Query String Features
def query_length(url):
    try: query = safe_urlparse(url).query; return len(query) if query else 0
    except: return 0
def num_query_params(url):
    try: query = safe_urlparse(url).query; return len(parse_qs(query)) if query else 0
    except: return 0

# 6. Lexical Keyword Features
SUSPICIOUS_KEYWORDS = ['login', 'secure', 'account', 'update', 'verify', 'signin', 'admin', 'banking', 'confirm', 'password', 'credential', 'support', 'service', 'official', 'webscr', 'cmd']
def count_suspicious_keywords(url):
    count = 0; url_lower = url.lower()
    for keyword in SUSPICIOUS_KEYWORDS:
        if keyword in url_lower: count += 1
    return count
def tld_in_subdomain_or_path(url):
    try:
        parsed = safe_urlparse(url); hostname = parsed.hostname or ''; path_query = parsed.path + ('?' + parsed.query if parsed.query else '')
        common_tlds = ['.com', '.org', '.net', '.gov', '.edu', '.info', '.biz', '.io', '.co', '.ru', '.cn', '.uk', '.de', '.jp']
        parts = hostname.split('.'); subdomain_part = '.'.join(parts[:-2])
        for tld in common_tlds:
            if tld in subdomain_part.lower(): return 1
            if tld in path_query.lower(): return 1
        return 0
    except Exception: return 0

# --- Function to extract ALL 29 features for the new model ---
def extract_url_features(url):
    """Extracts all 29 URL features for a single URL and returns a DataFrame row."""
    if not isinstance(url, str): # Handle potential non-string input
        url = ''

    features = {
        'having_ip': [having_ip_address(url)],
        'url_length_category': [url_length_category(url)], # Corrected function name
        'uses_https': [uses_https(url)],
        'has_at': [having_at_symbol(url)],
        'dot_count': [count_dots(url)],
        'hyphen_count': [count_hyphens(url)],
        'slash_count': [count_slashes(url)],
        'qmark_count': [count_questionmarks(url)],
        'equal_count': [count_equals(url)],
        'percent_count': [count_percent(url)],
        'underscore_count': [count_underscore(url)],
        'digit_count': [digit_count(url)],
        'letter_count': [letter_count(url)],
        'domain_length': [domain_length(url)],
        'prefix_suffix': [prefix_suffix(url)],
        'subdomain_count': [count_subdomains(url)],
        'https_token_domain': [https_token_in_domain(url)],
        'domain_entropy': [domain_entropy(url)],
        'domain_digit_ratio': [domain_digit_ratio(url)],
        'domain_letter_ratio': [domain_letter_ratio(url)],
        'path_length': [path_length(url)],
        'path_double_slash': [double_slash_redirecting(url)], # Use consistent name
        'path_entropy': [path_entropy(url)],
        'path_digit_ratio': [path_digit_ratio(url)],
        'path_letter_ratio': [path_letter_ratio(url)],
        'query_length': [query_length(url)],
        'num_query_params': [num_query_params(url)],
        'suspicious_keywords_count': [count_suspicious_keywords(url)],
        'tld_in_path_subdomain': [tld_in_subdomain_or_path(url)] # Use consistent name
    }
    # Ensure columns are in the EXACT same order as during training
    feature_order = [
        'having_ip', 'url_length_category', 'uses_https', 'has_at', 'dot_count',
        'hyphen_count', 'slash_count', 'qmark_count', 'equal_count', 'percent_count',
        'underscore_count', 'digit_count', 'letter_count', 'domain_length', 'prefix_suffix',
        'subdomain_count', 'https_token_domain', 'domain_entropy', 'domain_digit_ratio',
        'domain_letter_ratio', 'path_length', 'path_double_slash', 'path_entropy',
        'path_digit_ratio', 'path_letter_ratio', 'query_length', 'num_query_params',
        'suspicious_keywords_count', 'tld_in_path_subdomain'
        ]
    return pd.DataFrame(features)[feature_order]


# --- NLP Features ---
def preprocess_text(text):
    """Cleans text for NLP model."""
    if not isinstance(text, str): return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Add try-except for word_tokenize as it might fail on unusual input
    try:
        tokens = word_tokenize(text)
    except Exception as e:
        print(f" [Warning] NLTK word_tokenize failed: {e}")
        tokens = text.split() # Fallback to simple split

    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(filtered_tokens)

# --- CV Features ---
def preprocess_image(image_path, target_size=(224, 224)):
    """Loads and preprocesses an image for the CV model."""
    if not image_path or not os.path.exists(image_path):
         print(f"  [Error] Screenshot path invalid or file missing: {image_path}")
         return None
    try:
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0) # Add batch dimension
        return mobilenet_preprocess(img_array_expanded) # Use MobileNetV2 preprocessing
    except Exception as e:
        print(f"  [Error] Could not process image {image_path}: {e}")
        return None


# --- Prediction Function (Unchanged, relies on updated extract_url_features) ---
def get_predictions(url, page_text, screenshot_path):
    """
    Takes URL, text, and screenshot path, extracts features,
    and gets probability predictions from each model.
    Returns a dictionary of probabilities.
    """
    predictions = { "url_prob": 0.0, "nlp_prob": 0.0, "cv_prob": 0.0 }

    # 1. URL Model Prediction (Uses the updated extract_url_features)
    try:
        url_features_df = extract_url_features(url)
        url_pred_proba = url_model.predict_proba(url_features_df)
        predictions["url_prob"] = url_pred_proba[0][1]
        print(f"  URL Model Prediction (Prob Phishing): {predictions['url_prob']:.4f}")
    except Exception as e:
        print(f"  [Error] URL model prediction failed: {e}")

    # 2. NLP Model Prediction
    try:
        cleaned_text = preprocess_text(page_text)
        if cleaned_text and cleaned_text.strip(): # Check if not just whitespace
            text_tfidf = tfidf_vectorizer.transform([cleaned_text])
            nlp_pred_proba = nlp_model.predict_proba(text_tfidf)
            predictions["nlp_prob"] = nlp_pred_proba[0][1]
            print(f"  NLP Model Prediction (Prob Phishing): {predictions['nlp_prob']:.4f}")
        else:
             print("  NLP Model: No significant text content to analyze.")
    except Exception as e:
        print(f"  [Error] NLP model prediction failed: {e}")

    # 3. CV Model Prediction
    try:
        processed_image = preprocess_image(screenshot_path)
        if processed_image is not None:
            # Ensure batch dimension if needed (Keras predict usually handles it)
            cv_pred_proba = cv_model.predict(processed_image, verbose=0) # verbose=0 quieter
            predictions["cv_prob"] = cv_pred_proba[0][0]
            print(f"  CV Model Prediction (Prob Phishing): {predictions['cv_prob']:.4f}")
        else:
            print("  CV Model: Could not process screenshot.")
    except Exception as e:
        print(f"  [Error] CV model prediction failed: {e}")

    return predictions


# --- Example Usage (Optional - for testing this script directly) ---
if __name__ == "__main__":
    print("\n--- Testing Updated Prediction Script ---")
    try:
        # Load a sample from the dataset
        df_test_full = pd.read_csv("master_dataset.csv").dropna()
        if df_test_full.empty:
            print("[TEST FAILED] No valid rows in master_dataset.csv")
            exit()
        df_test = df_test_full.sample(1).iloc[0]
        test_url = df_test['url']
        test_text = df_test['page_text']
        test_screenshot = df_test['screenshot_path']
        actual_label = int(df_test['label']) # Ensure label is integer

        print(f"\nTesting with sample URL: {test_url}")
        print(f"(Actual Label: {'Phishing' if actual_label == 1 else 'Legitimate'})")
        print(f"Using text: '{test_text[:100]}...'")
        print(f"Using screenshot: {test_screenshot}")

        if not os.path.exists(test_screenshot):
             print(f"[WARNING] Screenshot file '{test_screenshot}' not found.")

        start_time = time.time()
        individual_preds = get_predictions(test_url, test_text, test_screenshot)
        end_time = time.time()

        print("\n--- Individual Model Probabilities (Phishing) ---")
        print(f"URL Model (XGB): {individual_preds['url_prob']:.4f}")
        print(f"NLP Model      : {individual_preds['nlp_prob']:.4f}")
        print(f"CV Model       : {individual_preds['cv_prob']:.4f}")
        print(f"\nPrediction time: {end_time - start_time:.2f} seconds")

    except FileNotFoundError: print("\n[TEST FAILED] master_dataset.csv not found."); exit()
    except ImportError as e: print(f"\n[TEST FAILED] Import Error: {e}"); exit()
    except Exception as e: print(f"\n[TEST FAILED] An error occurred: {e}"); exit()
    