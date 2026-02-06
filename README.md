# 🛡️ Multi-Modal Phishing Detection System

### *Real-Time Cyber-Threat Intelligence using XGBoost, NLP, and Computer Vision*

## 📌 Project Overview

Traditional phishing filters rely on static blocklists, which fail against "Zero-Hour" attacks. This project implements a **proactive, multi-modal detection engine** that analyzes a website's URL structure, textual content, and visual appearance in real-time.

By combining three specialized models into a **Stacked Ensemble**, the system achieves a final accuracy of **99.1%**, significantly reducing false positives by requiring a majority consensus.

---

## 🧠 The "Team of Experts" Architecture

The core of this system is its **Ensemble Strategy**. No single model makes the final decision; instead, they act as a team:

| Expert Model | Input Type | Algorithm | Core Task |
| --- | --- | --- | --- |
| **URL Analyst** | Raw URL | **XGBoost** | Analyzes 29 lexical features (entropy, length, subdomains). |
| **Content Analyst** | HTML Text | **Naive Bayes** | Processes page text using TF-IDF to find "bait" keywords. |
| **Visual Analyst** | Screenshot | **MobileNetV2** | Scans for visual clones of popular login pages (Google, Amazon). |

> **The Meta-Model:** A final **Logistic Regression** layer (the "Manager") weights the probabilities from each expert to output the final "Phishing" or "Legitimate" label.

---

## 🚀 Key Features

* **Live Scraping:** Uses **Selenium (Headless Chrome)** and **BeautifulSoup** to capture real-time data from any URL.
* **Ensemble Robustness:** Successfully handles "False Positives" (e.g., complex Amazon ad-tracking URLs) by using majority voting.
* **Cross-Platform:** Includes a **Flask Web Interface** for manual testing and a **JSON API** for integration.
* **Production Ready:** Built with **CORS** support for seamless interaction with browser extensions.

---

## 🛠️ Technical Stack

* **Backend:** Flask (Python)
* **Machine Learning:** Scikit-Learn, XGBoost, TensorFlow (Keras)
* **Automation:** Selenium WebDriver (Chrome)
* **NLP:** NLTK (Stopword removal, Tokenization)
* **Data Handling:** Pandas, NumPy, Joblib

---

## 📂 Project Structure

```text
├── app.py                # Flask Server (API & Web Interface)
├── predict_phishing.py   # The Prediction Engine (Connects all models)
├── train_meta_model.py   # Training script for the Stacking Ensemble
├── models/               # Saved .joblib and .keras model files
├── templates/            # HTML Web Interface (index.html)
└── temp_screenshots/     # Temporary storage for live URL analysis

```

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/phishing-engine.git
cd phishing-engine

```


2. **Create a virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```


4. **Run the Server:**
```bash
python app.py

```


*Access the web interface at: `http://127.0.0.1:5000*`

---

## 📈 Model Performance

Based on recent training logs:

* **URL Model Accuracy:** 99.1%
* **NLP Model Accuracy:** 89.1%
* **Final Meta-Model Accuracy:** **99.1%**

---



---

### Would you like me to help you write a "Conclusion" section for your report based on these results next?
