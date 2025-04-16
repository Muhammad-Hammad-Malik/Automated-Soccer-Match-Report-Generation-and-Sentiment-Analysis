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

# === FUNCTION TO SCRAPE ONE LINK AND DIVIDE BY MINUTES ===
def scrape_commentary(url):
    try:
        driver.get(url)

        # Click "Show more" button if available
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
            pass  # Button might not exist

        # Scrape the commentary
        content_xpath = "/html/body/div[1]/main/div[2]/div[2]/div[1]/div[1]/div[7]/div[4]"
        content_div = driver.find_element(By.XPATH, content_xpath)
        lines = content_div.text.strip().split('\n')

        # Group commentary pairs (minute + description)
        events = []
        i = 0
        while i < len(lines) - 1:
            if lines[i].endswith("'"):
                minute_text = lines[i].strip()
                desc = lines[i + 1].strip()
                try:
                    minute = int(minute_text.replace("'", "").replace("+", "").strip())
                    events.append((minute, desc))  # Strip the minute
                except:
                    pass
                i += 2
            else:
                i += 1

        # Sort from earliest to latest
        events.reverse()

        # Split into parts
        def get_part_text(min_start, min_end=None):
            part = [desc for min, desc in events if min >= min_start and (min_end is None or min < min_end)]
            return " ".join(part).strip()

        part_1 = get_part_text(0, 25)
        part_2 = get_part_text(25, 45)
        part_3 = get_part_text(45, 70)
        part_4 = get_part_text(70)

        return [part_1, part_2, part_3, part_4]

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

# === READ LINKS AND SCRAPE ===
if not os.path.exists(LINKS_FILE):
    print(f"'{LINKS_FILE}' not found!")
else:
    with open(LINKS_FILE, "r", encoding="utf-8") as f:
        links = [line.strip() for line in f if line.strip()]

    # Prepare CSV file
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, "a", newline='', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Commentary"])
        if not file_exists:
            writer.writeheader()

        for link in links:
            print(f"Scraping: {link}")
            parts = scrape_commentary(link)
            if parts:
                for part in parts:
                    if part.strip():  # Only write non-empty parts
                        writer.writerow({"Commentary": part.strip()})
                print("✔ Parts written to CSV\n")
            else:
                print("✖ Skipped (no data)\n")

# === CLEANUP ===
driver.quit()
print("✅ Finished scraping and writing to CSV.")
