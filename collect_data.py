import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import os
import time
import random

# --- CONFIGURATION ---
NUM_LEGITIMATE_SAMPLES = 2000  # Number of "good" sites to scrape
NUM_PHISHING_SAMPLES = 2000     # Number of "bad" sites to scrape
# --- RESTART LOGIC ---
RESTART_DRIVER_EVERY = 100     # How many sites before restarting browser
# ---------------------
OUTPUT_CSV = "master_dataset.csv"
SCREENSHOT_DIR = "screenshots"
# ---------------------

def setup_driver():
    """Sets up the Selenium WebDriver with headless options."""
    print("  Setting up new WebDriver session...")
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")

    s = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=s, options=chrome_options)
    # Set timeout on the driver object itself
    driver.set_page_load_timeout(20)
    print("  WebDriver session started.")
    return driver

def scrape_site_data(url, driver, label):
    """
    Visits a URL, scrapes its text, and takes a screenshot.
    Returns a dictionary of the collected data.
    """
    try:
        driver.get(url)
        time.sleep(1) # Reduced wait time

        html = driver.page_source
        soup = BeautifulSoup(html, 'lxml') # Faster parser

        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        page_text = ' '.join(soup.stripped_strings)

        # Handle cases where page text might be empty even after successful load
        if not page_text:
             print(f"  [Warning] Empty page text for {url}")
             page_text = " " # Assign a space to avoid issues later

        filename = f"{label}_{random.randint(10000, 99999)}.png"
        filepath = os.path.join(SCREENSHOT_DIR, filename)
        driver.save_screenshot(filepath)

        return {
            "url": url,
            "page_text": page_text,
            "screenshot_path": filepath,
            "label": label,
            "status": "success"
        }
    except Exception as e:
        print(f"  [Error] Failed to scrape {url}: {type(e).__name__} - {e}")
        # Check if it's the specific invalid session error, potentially trigger restart sooner if needed
        # (For now, we just rely on the periodic restart)
        return {
            "url": url,
            "page_text": None,
            "screenshot_path": None,
            "label": label,
            "status": "error"
        }

def main():
    print("--- Phase 1: Data Collection Script (with Auto-Restart) ---")

    if not os.path.exists(SCREENSHOT_DIR):
        os.makedirs(SCREENSHOT_DIR)
        print(f"Created directory: {SCREENSHOT_DIR}")

    print("Loading URL lists...")
    try:
        tranco_df = pd.read_csv("tranco_top1m.csv", names=["rank", "url"])
    except FileNotFoundError:
        print("[ERROR] tranco_top1m.csv not found!")
        return

    legit_urls = tranco_df.sample(NUM_LEGITIMATE_SAMPLES)['url'].tolist()
    legit_urls = ["http://" + url for url in legit_urls]
    legit_data = [{"url": url, "label": 0} for url in legit_urls]

    try:
        phishtank_df = pd.read_csv("verified_online.csv")
    except FileNotFoundError:
        print("[ERROR] verified_online.csv not found!")
        return

    phish_urls = phishtank_df.sample(NUM_PHISHING_SAMPLES)['url'].tolist()
    phish_data = [{"url": url, "label": 1} for url in phish_urls]

    all_urls_to_scrape = legit_data + phish_data
    random.shuffle(all_urls_to_scrape)
    total_sites = len(all_urls_to_scrape)
    print(f"Loaded {NUM_LEGITIMATE_SAMPLES} legitimate and {NUM_PHISHING_SAMPLES} phishing URLs.")
    print(f"Total sites to scrape: {total_sites}")

    all_results = []
    driver = None # Initialize driver as None

    print("\nStarting scraper... This will take some time.")

    for i, site_info in enumerate(all_urls_to_scrape):
        # --- RESTART LOGIC ---
        # Check if we need to start/restart the driver
        if i % RESTART_DRIVER_EVERY == 0:
            if driver is not None:
                print(f"\n--- Restarting WebDriver (after {i} sites) ---")
                driver.quit()
            driver = setup_driver()
        # ---------------------

        print(f"Scraping {i+1}/{total_sites}: {site_info['url']}")
        # --- Make sure driver is valid before scraping ---
        if driver:
             result = scrape_site_data(site_info['url'], driver, site_info['label'])
             all_results.append(result)
        else:
             print("  [Error] WebDriver not initialized, skipping.")
             all_results.append({
                 "url": site_info['url'],
                 "page_text": None,
                 "screenshot_path": None,
                 "label": site_info['label'],
                 "status": "error_driver_init"
             })
        # --------------------------------------------------

    # --- Final Quit ---
    if driver is not None:
        print("\nScraping complete. Shutting down final driver session.")
        driver.quit()
    # ------------------

    final_dataset = pd.DataFrame(all_results)
    successful_scrapes = final_dataset[final_dataset['status'] == 'success'].copy()

    if successful_scrapes.empty:
         print("[ERROR] No sites were scraped successfully.")
    else:
        successful_scrapes = successful_scrapes[['url', 'page_text', 'screenshot_path', 'label']]
        successful_scrapes.to_csv(OUTPUT_CSV, index=False)

        print("--- SCRIPT FINISHED ---")
        print(f"Successfully scraped {len(successful_scrapes)} sites.")
        print(f"Master dataset saved to: {OUTPUT_CSV}")
        print(f"Screenshots saved in: {SCREENSHOT_DIR}")

if __name__ == "__main__":
    main()