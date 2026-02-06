# --- START of app.py ---

# --- Core Flask and Utilities ---
from flask import Flask, request, render_template, redirect, url_for, jsonify
from flask_cors import CORS # Import CORS
import joblib
import pandas as pd
import numpy as np
import os
import time
import random
from urllib.parse import urlparse

# --- Import prediction logic and models ---
try:
    print("Loading base models and vectorizer...")
    from predict_phishing import (
        extract_url_features,
        preprocess_text,
        preprocess_image,
        url_model,
        nlp_model,
        tfidf_vectorizer,
        cv_model
    )
    print("Base models (URL, NLP, CV) and vectorizer loaded successfully.")
except FileNotFoundError as e:
    print(f"[FATAL ERROR] Model or vectorizer file not found: {e}.")
    exit()
except ImportError as e:
    print(f"[FATAL ERROR] Could not import from predict_phishing.py: {e}")
    exit()
except Exception as e:
    print(f"[FATAL ERROR] Failed to load models/vectorizer: {e}")
    exit()


# --- Live Scraping Function ---
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager
    from bs4 import BeautifulSoup
    print("Selenium and BeautifulSoup imported successfully.")
except ImportError:
    print("[FATAL ERROR] Missing libraries. Run: pip install Flask Flask-Cors selenium webdriver-manager beautifulsoup4 lxml")
    exit()


TEMP_SCREENSHOT_DIR = "temp_screenshots"

def setup_live_driver():
    """Sets up Selenium WebDriver for live scraping."""
    # ... (Keep the setup_live_driver function as it was in the previous correct version)
    if not os.path.exists(TEMP_SCREENSHOT_DIR):
        try: os.makedirs(TEMP_SCREENSHOT_DIR); print(f"Created temp dir: {TEMP_SCREENSHOT_DIR}")
        except OSError as e: print(f"[ERROR] Cannot create temp dir {TEMP_SCREENSHOT_DIR}: {e}"); return None

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36")
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])

    try:
        print("Setting up WebDriver via Manager...")
        os.environ['WDM_LOG'] = '0'; os.environ['WDM_LOG_LEVEL'] = '0' # Suppress WDM logs
        s = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=s, options=chrome_options)
        driver.set_page_load_timeout(25)
        print("WebDriver setup complete.")
        return driver
    except Exception as e:
        print(f"[ERROR] WebDriver Manager failed: {e}")
        try:
            print("Trying WebDriver setup without explicit manager...")
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(25)
            print("WebDriver setup complete (no manager).")
            return driver
        except Exception as e2:
            print(f"[FATAL ERROR] WebDriver setup failed completely: {e2}"); return None


def scrape_live_data(url_to_scrape):
    """Scrapes text and takes screenshot for a live URL."""
    # ... (Keep the scrape_live_data function as it was, ensuring it uses try/except for os.remove)
    driver = None
    if not os.path.exists(TEMP_SCREENSHOT_DIR): os.makedirs(TEMP_SCREENSHOT_DIR, exist_ok=True)
    scrape_result = {"page_text": " ", "screenshot_path": None, "error": None}
    screenshot_path_temp = os.path.join(TEMP_SCREENSHOT_DIR, f"live_{random.randint(1000,9999)}.png")

    try:
        driver = setup_live_driver()
        if driver is None: raise Exception("WebDriver init failed.")
        print(f"Attempting to scrape live URL: {url_to_scrape}")
        driver.get(url_to_scrape)
        time.sleep(1.5)
        html = driver.page_source
        if html:
            soup = BeautifulSoup(html, 'lxml')
            for tag in soup(["script", "style", "header", "footer", "nav", "aside"]): tag.decompose()
            page_text = ' '.join(soup.stripped_strings); scrape_result["page_text"] = page_text if page_text else " "
        else: print("  [Warning] Page source empty.")
        driver.save_screenshot(screenshot_path_temp)
        scrape_result["screenshot_path"] = screenshot_path_temp
        print(f"Live scrape successful. Screenshot: {screenshot_path_temp}")
    except Exception as e:
        error_message = f"{type(e).__name__} - {e}"; print(f"[ERROR] Live scraping failed: {error_message}")
        scrape_result["error"] = error_message
        if os.path.exists(screenshot_path_temp): # Clean up partial success
            try: os.remove(screenshot_path_temp)
            except Exception as del_e: print(f"  [Warning] Failed to delete temp screenshot {screenshot_path_temp}: {del_e}")
        scrape_result["screenshot_path"] = None
    finally:
        if driver: driver.quit(); print("Live driver session closed.")
    return scrape_result

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app)

# --- Define Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        url_to_check = request.form.get('url', '').strip()
        print(f"\nWeb request received for URL: {url_to_check}")
        # --- Input Validation ---
        if not url_to_check: return render_template('index.html', result="Please enter a URL.")
        if not url_to_check.startswith(('http://', 'https://')): url_to_check = "http://" + url_to_check
        try: parsed = urlparse(url_to_check); assert parsed.scheme and parsed.netloc
        except: return render_template('index.html', result="Invalid URL structure.")

        # 1. Scrape Live Data
        scraped_data = scrape_live_data(url_to_check)
        screenshot_path = scraped_data.get("screenshot_path")

        # 2. Handle Scraping Failure
        if scraped_data["error"] or not scraped_data["page_text"] or not screenshot_path:
            result_message = f"Could not analyze URL. Scraping failed: {scraped_data.get('error', 'Unknown')}"
            # --- FIXED SYNTAX HERE ---
            if screenshot_path and os.path.exists(screenshot_path):
                try: os.remove(screenshot_path)
                except Exception as del_e: print(f"  [Warning] Predict: Could not delete temp screenshot on failure: {del_e}")
            # -------------------------
            return render_template('index.html', result=result_message)

        page_text = scraped_data["page_text"]

        # 3. Get Base Model Predictions (Keep as before)
        print("Getting predictions from base models...")
        base_model_probs = { "url_prob": 0.0, "nlp_prob": 0.0, "cv_prob": 0.0 }
        prediction_error = False
        try:
            url_features = extract_url_features(url_to_check)
            base_model_probs["url_prob"] = url_model.predict_proba(url_features)[0][1]

            cleaned_text = preprocess_text(page_text)
            if cleaned_text and cleaned_text.strip():
                text_tfidf = tfidf_vectorizer.transform([cleaned_text])
                base_model_probs["nlp_prob"] = nlp_model.predict_proba(text_tfidf)[0][1]

            processed_image = preprocess_image(screenshot_path)
            if processed_image is not None:
                if processed_image.ndim == 3: processed_image = np.expand_dims(processed_image, axis=0)
                base_model_probs["cv_prob"] = cv_model.predict(processed_image, verbose=0)[0][0]

            print(f"Base model probabilities: {base_model_probs}")
        except Exception as e:
            print(f"[ERROR] Failed during base model prediction: {e}"); prediction_error = True

        # 4. Apply Majority Vote (Keep as before)
        result_message = "Error during prediction."
        prediction_label = -1
        if not prediction_error:
            print("Applying Majority Vote...")
            try:
                url_pred = 1 if base_model_probs["url_prob"] > 0.5 else 0
                nlp_pred = 1 if base_model_probs["nlp_prob"] > 0.5 else 0
                cv_pred  = 1 if base_model_probs["cv_prob"] > 0.5 else 0
                phishing_votes = url_pred + nlp_pred + cv_pred
                print(f"Votes - URL:{url_pred}({base_model_probs['url_prob']:.2f}), NLP:{nlp_pred}({base_model_probs['nlp_prob']:.2f}), CV:{cv_pred}({base_model_probs['cv_prob']:.2f}) -> Total:{phishing_votes}")
                if phishing_votes >= 2: result_message = f"Warning: This URL appears to be PHISHING! (Vote: {phishing_votes}/3)"; prediction_label = 1
                else: result_message = f"This URL appears LEGITIMATE. (Vote: {phishing_votes}/3)"; prediction_label = 0
            except Exception as e: print(f"[ERROR] Failed during majority vote: {e}"); result_message="Error during vote."; prediction_label=-1
        else: result_message = "Prediction failed due to error in base models."

        # 5. Clean up Screenshot
        if screenshot_path and os.path.exists(screenshot_path):
            try: os.remove(screenshot_path); print(f"Temp screenshot {screenshot_path} deleted.")
            # --- FIXED SYNTAX HERE ---
            except Exception as del_e: print(f"  [Warning] Predict: Failed to delete temp screenshot {screenshot_path}: {del_e}")
            # -------------------------

        # 6. Render Result Page
        return render_template('index.html', result=result_message, prediction=prediction_label, checked_url=url_to_check)

    return redirect(url_for('home'))


# --- API Endpoint for Browser Extension ---
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Handles URL submission from browser extension, returns JSON."""
    if request.method == 'POST':
        url_to_check = request.form.get('url', '').strip()
        print(f"\nAPI request received for URL: {url_to_check}")
        # --- Input Validation ---
        if not url_to_check: return jsonify({"error": "No URL provided"}), 400
        if not url_to_check.startswith(('http://', 'https://')): url_to_check = "http://" + url_to_check
        try: parsed = urlparse(url_to_check); assert parsed.scheme and parsed.netloc
        except: return jsonify({"error": "Invalid URL structure"}), 400

        # 1. Scrape Live Data
        scraped_data = scrape_live_data(url_to_check)
        screenshot_path = scraped_data.get("screenshot_path")

        # 2. Handle Scraping Failure
        if scraped_data["error"] or not scraped_data["page_text"] or not screenshot_path:
            # --- FIXED SYNTAX HERE ---
            if screenshot_path and os.path.exists(screenshot_path):
                try: os.remove(screenshot_path)
                except Exception as del_e: print(f"  [Warning] API: Could not delete temp screenshot on failure: {del_e}")
            # -------------------------
            return jsonify({"error": f"Scraping failed: {scraped_data.get('error', 'Unknown')}", "prediction": -1}), 500

        page_text = scraped_data["page_text"]

        # 3. Get Base Model Predictions (Keep as before)
        base_model_probs = { "url_prob": 0.0, "nlp_prob": 0.0, "cv_prob": 0.0 }
        prediction_error = False
        try:
            url_features = extract_url_features(url_to_check)
            base_model_probs["url_prob"] = url_model.predict_proba(url_features)[0][1]
            cleaned_text = preprocess_text(page_text)
            if cleaned_text and cleaned_text.strip():
                text_tfidf = tfidf_vectorizer.transform([cleaned_text])
                base_model_probs["nlp_prob"] = nlp_model.predict_proba(text_tfidf)[0][1]
            processed_image = preprocess_image(screenshot_path)
            if processed_image is not None:
                if processed_image.ndim == 3: processed_image = np.expand_dims(processed_image, axis=0)
                base_model_probs["cv_prob"] = cv_model.predict(processed_image, verbose=0)[0][0]
            print(f"API Base model probabilities: {base_model_probs}")
        except Exception as e: print(f"[ERROR] API: Base model prediction failed: {e}"); prediction_error = True

        # 4. Apply Majority Vote (Keep as before)
        final_prediction = -1
        if not prediction_error:
            print("API: Applying Majority Vote...")
            try:
                url_pred = 1 if base_model_probs["url_prob"] > 0.5 else 0
                nlp_pred = 1 if base_model_probs["nlp_prob"] > 0.5 else 0
                cv_pred  = 1 if base_model_probs["cv_prob"] > 0.5 else 0
                phishing_votes = url_pred + nlp_pred + cv_pred
                print(f"API Votes - URL:{url_pred}({base_model_probs['url_prob']:.2f}), NLP:{nlp_pred}({base_model_probs['nlp_prob']:.2f}), CV:{cv_pred}({base_model_probs['cv_prob']:.2f}) -> Total:{phishing_votes}")
                final_prediction = 1 if phishing_votes >= 2 else 0
            except Exception as e: print(f"[ERROR] API: Majority vote failed: {e}"); final_prediction = -1
        else: final_prediction = -1

        # 5. Clean up Screenshot
        if screenshot_path and os.path.exists(screenshot_path):
            try: os.remove(screenshot_path); print(f"API: Temp screenshot {screenshot_path} deleted.")
            # --- FIXED SYNTAX HERE ---
            except Exception as del_e: print(f"  [Warning] API: Failed to delete temp screenshot {screenshot_path}: {del_e}")
            # -------------------------

        # 6. Return JSON response
        if final_prediction != -1: print(f"API sending prediction: {final_prediction}"); return jsonify({"prediction": final_prediction, "url": url_to_check})
        else: print("API sending error response"); return jsonify({"error": "Prediction failed", "prediction": -1}), 500

    return jsonify({"error": "Method Not Allowed - Use POST"}), 405


# --- Run the Flask App ---
if __name__ == '__main__':
    # Ensure NLTK is available
    try:
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        print("Checking NLTK data...")
        _ = stopwords.words('english')
        _ = word_tokenize("test")
        print("NLTK data found.")
    except ImportError: print("[ERROR] NLTK missing."); exit()
    except LookupError:
        print("\n[NLTK Download Needed] Attempting download...")
        try:
            import nltk
            nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)
            print("NLTK data downloaded."); _ = stopwords.words('english'); _ = word_tokenize("test")
        except Exception as e:
            print(f"[ERROR] Failed NLTK download: {e}"); print("Run manually: import nltk; nltk.download('punkt'); nltk.download('stopwords')"); exit()

    # Check for Flask-CORS
    try: import flask_cors; print("Flask-CORS found.")
    except ImportError: print("[ERROR] Flask-CORS missing. Run: pip install Flask-Cors"); exit()

    print("\nStarting Flask development server...")
    print(" >>> Access the web interface at: http://127.0.0.1:5000/ <<< ")
    print(" >>> The browser extension should send requests to: http://127.0.0.1:5000/api/predict <<< ")
    app.run(debug=True, host='0.0.0.0', port=5000)

# --- END of app.py ---54