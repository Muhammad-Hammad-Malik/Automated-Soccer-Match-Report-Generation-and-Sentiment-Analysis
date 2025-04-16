import os
import csv
import time
import warnings
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# === SUPPRESS WARNINGS ===
warnings.filterwarnings("ignore")

# === CONFIGURATION ===
CHROMEDRIVER_PATH = "chromedriver.exe"
LINKS_FILE = "links.txt"
CSV_FILE = "commentary_dataset.csv"

# === SETUP CHROME OPTIONS (HEADLESS) ===
options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

service = Service(executable_path=CHROMEDRIVER_PATH)
driver = webdriver.Chrome(service=service, options=options)
wait = WebDriverWait(driver, 20)

# === FUNCTION TO SCRAPE ONE LINK ===
def scrape_commentary(url):
    try:
        driver.get(url)

        # Scroll to and click "Show more" button if available
        button_selector = "button.button--variant_clear.button--size_secondary"
        try:
            button = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, button_selector)))
            driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", button)
            time.sleep(1)
            try:
                wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, button_selector)))
                button.click()
            except:
                driver.execute_script("arguments[0].click();", button)
            time.sleep(2)
        except:
            pass  # Button might not exist on some pages

        # Scrape the commentary
        content_xpath = "/html/body/div[1]/main/div[2]/div[2]/div[1]/div[1]/div[7]/div[4]"
        content_div = driver.find_element(By.XPATH, content_xpath)
        raw_text = content_div.text

        # Process and clean the text
        lines = raw_text.strip().split('\n')
        cleaned_events = []
        for line in lines:
            if not line.endswith("'") or not line[:-1].isdigit():
                cleaned_events.append(line.strip())

        # Reverse to get oldest-to-newest order
        cleaned_events.reverse()
        paragraph = " ".join(cleaned_events)

        return paragraph

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

# === READ LINKS AND SCRAPE ===
if not os.path.exists(LINKS_FILE):
    print(f"'{LINKS_FILE}' not found!")
else:
    with open(LINKS_FILE, "r", encoding="utf-8") as f:
        links = [line.strip() for line in f if line.strip()]

    # Prepare to write to CSV
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, "a", newline='', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Commentary"])

        if not file_exists:
            writer.writeheader()

        for link in links:
            print(f"Scraping: {link}")
            commentary = scrape_commentary(link)
            if commentary:
                writer.writerow({"Commentary": commentary})
                print("✔ Added to CSV\n")
            else:
                print("✖ Skipped (no data)\n")

# === CLEANUP ===
driver.quit()
print("✅ Finished scraping and saving to CSV.")
