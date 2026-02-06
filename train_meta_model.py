import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # Good choice for a meta-model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# --- Import functions from predict_phishing ---
# Make sure predict_phishing.py is in the same folder
try:
    from predict_phishing import extract_url_features, preprocess_text, preprocess_image, url_model, nlp_model, tfidf_vectorizer, cv_model
    print("Successfully imported models and functions from predict_phishing.py")
except ImportError as e:
    print(f"[ERROR] Could not import from predict_phishing.py: {e}")
    print("Ensure predict_phishing.py is in the same directory and all models load correctly.")
    exit()
except Exception as e:
    print(f"[ERROR] An error occurred during import from predict_phishing.py: {e}")
    exit()

# --- Configuration ---
DATASET_PATH = "master_dataset.csv"
META_MODEL_SAVE_PATH = "meta_model.joblib"
# Use the same random state and test size as previous phases for consistency
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Main Meta-Model Training Script ---
def main():
    print("\n--- Phase 5: Training the Meta-Model ---")

    # 1. Load the full dataset
    print("Loading full dataset...")
    try:
        df = pd.read_csv(DATASET_PATH)
        df = df.dropna(subset=['url', 'page_text', 'screenshot_path', 'label']) # Use only complete rows
        print(f"Loaded {len(df)} complete rows.")
    except FileNotFoundError:
        print(f"[ERROR] Dataset not found at {DATASET_PATH}")
        return
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return

    # 2. Split data into Train and Test sets (using the same split as before)
    # We only need the test set URLs/paths to generate features for the meta-model
    print("Splitting data to get the test set indices...")
    # Create dummy features just for splitting indices correctly
    dummy_X = df[['url']] # Just need something with the right number of rows
    y = df['label']
    _, X_test_indices, _, y_test = train_test_split(dummy_X.index, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    print(f"Using {len(X_test_indices)} samples from the test set to generate meta-features.")
    test_df = df.loc[X_test_indices].copy() # Get the actual data for the test set

    # 3. Generate Predictions from Base Models (Meta-Features) for the Test Set
    print("\nGenerating predictions from base models for the test set (this might take a while)...")
    meta_features = []
    processed_count = 0
    for index, row in test_df.iterrows():
        url = row['url']
        text = row['page_text']
        screenshot = row['screenshot_path']
        actual_label = row['label'] # Keep track for final evaluation if needed

        # Ensure screenshot file exists before trying to predict
        if not os.path.exists(screenshot):
             print(f"  [Skipping row {index}] Screenshot not found: {screenshot}")
             continue # Skip this row if screenshot is missing

        # --- Get URL prediction ---
        url_prob = 0.0
        try:
            url_features_df = extract_url_features(url)
            url_pred_proba = url_model.predict_proba(url_features_df)
            url_prob = url_pred_proba[0][1]
        except Exception as e:
            print(f"  [Warning] URL model failed for {url}: {e}")

        # --- Get NLP prediction ---
        nlp_prob = 0.0
        try:
            cleaned_text = preprocess_text(text)
            if cleaned_text:
                text_tfidf = tfidf_vectorizer.transform([cleaned_text])
                nlp_pred_proba = nlp_model.predict_proba(text_tfidf)
                nlp_prob = nlp_pred_proba[0][1]
        except Exception as e:
            print(f"  [Warning] NLP model failed for {url}: {e}")

        # --- Get CV prediction ---
        cv_prob = 0.0
        try:
            processed_image = preprocess_image(screenshot)
            if processed_image is not None:
                cv_pred_proba = cv_model.predict(processed_image, verbose=0) # verbose=0 suppresses progress bar
                cv_prob = cv_pred_proba[0][0]
        except Exception as e:
            print(f"  [Warning] CV model failed for {url} ({screenshot}): {e}")

        meta_features.append({
            'url_prob': url_prob,
            'nlp_prob': nlp_prob,
            'cv_prob': cv_prob,
            'actual_label': actual_label # Include actual label for training
        })
        processed_count += 1
        if processed_count % 50 == 0: # Print progress update
             print(f"  Processed {processed_count}/{len(test_df)} test samples...")

    print(f"\nGenerated meta-features for {len(meta_features)} samples.")

    if not meta_features:
        print("[ERROR] No meta-features were generated. Check for errors in base model predictions or data paths.")
        return

    # 4. Prepare data for Meta-Model Training
    meta_df = pd.DataFrame(meta_features)
    X_meta = meta_df[['url_prob', 'nlp_prob', 'cv_prob']] # Features are the probabilities
    y_meta = meta_df['actual_label']                     # Target is the original label

    # 5. Train the Meta-Model (Logistic Regression)
    print("\nTraining the Meta-Model (Logistic Regression)...")
    meta_model = LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced')
    meta_model.fit(X_meta, y_meta)
    print("Meta-Model training complete.")

    # 6. Evaluate the Meta-Model (on the same data it was trained on - just for a quick check)
    # Ideally, you'd have a separate validation set for the meta-model,
    # but for simplicity, we'll evaluate on the test predictions.
    print("\nEvaluating Meta-Model performance (on test set predictions)...")
    y_meta_pred = meta_model.predict(X_meta)

    print("\nClassification Report (Meta-Model):")
    print(classification_report(y_meta, y_meta_pred, target_names=['Legitimate (0)', 'Phishing (1)']))

    print("\nConfusion Matrix (Meta-Model):")
    print(confusion_matrix(y_meta, y_meta_pred))

    print(f"\nOverall Accuracy (Meta-Model): {accuracy_score(y_meta, y_meta_pred):.4f}")

    # Display Coefficients (how much weight it gives each model)
    print("\nMeta-Model Coefficients (Importance Weights):")
    try:
        coefficients = meta_model.coef_[0]
        feature_names = X_meta.columns
        coeff_df = pd.DataFrame({'Model': feature_names, 'Coefficient': coefficients})
        coeff_df = coeff_df.sort_values(by='Coefficient', ascending=False)
        print(coeff_df)
        print("(Positive values mean higher probability increases phishing likelihood)")
    except Exception as e:
        print(f"Could not display coefficients: {e}")

    # 7. Save the Meta-Model
    print(f"\nSaving Meta-Model to {META_MODEL_SAVE_PATH}...")
    joblib.dump(meta_model, META_MODEL_SAVE_PATH)
    print("Meta-Model saved successfully.")

    print("\n--- Meta-Model Training Finished ---")


if __name__ == "__main__":
    main()