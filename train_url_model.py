import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier # Using XGBoost
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from urllib.parse import urlparse, parse_qs
import ipaddress
import math
# import whois # Removed
from datetime import datetime # Keep for potential future use, but not strictly needed now
import time
import string
from collections import Counter

# --- Feature Extraction Functions ---

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
SUSPICIOUS_KEYWORDS = ['login', 'secure', 'account', 'update', 'verify', 'signin', 'admin', 'banking', 'confirm', 'password', 'credential', 'support', 'service', 'official', 'webscr', 'cmd'] # Added cmd, webscr
def count_suspicious_keywords(url):
    count = 0; url_lower = url.lower()
    for keyword in SUSPICIOUS_KEYWORDS:
        if keyword in url_lower: count += 1
    return count
def tld_in_subdomain_or_path(url):
    try:
        parsed = safe_urlparse(url); hostname = parsed.hostname or ''; path_query = parsed.path + ('?' + parsed.query if parsed.query else '')
        common_tlds = ['.com', '.org', '.net', '.gov', '.edu', '.info', '.biz', '.io', '.co', '.ru', '.cn', '.uk', '.de', '.jp'] # Expanded slightly
        parts = hostname.split('.'); subdomain_part = '.'.join(parts[:-2])
        for tld in common_tlds:
            if tld in subdomain_part.lower(): return 1
            if tld in path_query.lower(): return 1
        return 0
    except Exception: return 0

# --- Domain Age Function REMOVED ---
# def get_domain_age(url): ...

# --- Main Training Script ---
def main():
    print("--- Phase 2 (Retrain): Training URL Model (No Domain Age) ---")

    # 1. Load Data
    print("Loading dataset...")
    try: df = pd.read_csv("master_dataset.csv").dropna(subset=['url', 'label']); print(f"Loaded {len(df)} rows.")
    except Exception as e: print(f"[ERROR] Load failed: {e}"); return

    # 2. Extract Features (Excluding Domain Age)
    print("Extracting URL features (NO Domain Age)...")
    features = pd.DataFrame()
    feature_functions = [
        having_ip_address, url_length_category, uses_https, having_at_symbol, count_dots,
        count_hyphens, count_slashes, count_questionmarks, count_equals, count_percent,
        count_underscore, digit_count, letter_count, domain_length, prefix_suffix,
        count_subdomains, https_token_in_domain, domain_entropy, domain_digit_ratio,
        domain_letter_ratio, path_length, double_slash_redirecting, path_entropy,
        path_digit_ratio, path_letter_ratio, query_length, num_query_params,
        count_suspicious_keywords, tld_in_subdomain_or_path # Removed get_domain_age
    ]
    feature_names = [
        'having_ip', 'url_length_category', 'uses_https', 'has_at', 'dot_count',
        'hyphen_count', 'slash_count', 'qmark_count', 'equal_count', 'percent_count',
        'underscore_count', 'digit_count', 'letter_count', 'domain_length', 'prefix_suffix',
        'subdomain_count', 'https_token_domain', 'domain_entropy', 'domain_digit_ratio',
        'domain_letter_ratio', 'path_length', 'path_double_slash', 'path_entropy',
        'path_digit_ratio', 'path_letter_ratio', 'query_length', 'num_query_params',
        'suspicious_keywords_count', 'tld_in_path_subdomain' # Removed domain_age_days
    ]

    if len(feature_functions) != len(feature_names): print("[ERROR] Mismatch functions/names!"); return

    total_funcs = len(feature_functions)
    for i, (func, name) in enumerate(zip(feature_functions, feature_names)):
        print(f"  Extracting feature {i+1}/{total_funcs}: {name}...")
        features[name] = df['url'].apply(lambda url: func(url) if isinstance(url, str) else 0)

    labels = df['label']
    X = features
    y = labels
    print("Features extracted:")
    print(X.head())
    print(f"\nTotal features extracted: {len(X.columns)}")

    # 3. Split Data
    print("\nSplitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # 4. Train XGBoost Model
    print("\nTraining XGBoost model (enhanced features, no domain age)...")
    model = XGBClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=7, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1,
        use_label_encoder=False, eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 5. Evaluate
    print("\nEvaluating model performance...")
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate (0)', 'Phishing (1)']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nOverall Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    print("\nFeature Importances (XGBoost):")
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
    print(feature_importance_df.sort_values(by='importance', ascending=False).to_string())

    # 6. Save Updated Model
    model_filename = "url_model_xgb_no_age.joblib" # New name indicating no domain age
    print(f"\nSaving updated XGBoost model to {model_filename}...")
    joblib.dump(model, model_filename)
    print("Model saved successfully.")
    print(f"\n--- Phase 2 (Retrain XGBoost No Age) Finished ---")
    print(f"\nIMPORTANT: Remember to update predict_phishing.py and potentially app.py")
    print(f"           to use '{model_filename}' and include all the *new* feature functions (excluding domain age)!")


if __name__ == "__main__":
    # Removed whois check
    main()