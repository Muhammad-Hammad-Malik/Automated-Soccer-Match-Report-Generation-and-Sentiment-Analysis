import os
import time
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# === SUPPRESS WARNINGS ===
import warnings
warnings.filterwarnings("ignore")

# Optionally silence stdout/stderr from external libs (e.g., Chrome/DevTools junk)
import sys
class DevNull:
    def write(self, msg): pass
    def flush(self): pass

# Redirect only stderr to null (keep stdout for real output)
sys.stderr = DevNull()
# === CONFIGURATION ===
CHROMEDRIVER_PATH = "chromedriver.exe"
MODEL_PATH = "match_report_model.pt"
DEVICE = torch.device("cpu")
MODEL_NAME = "google/flan-t5-small"  # or whatever FLAN model you used

# === SETUP CHROME OPTIONS (HEADLESS) ===
options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
service = Service(executable_path=CHROMEDRIVER_PATH)
driver = webdriver.Chrome(service=service, options=options)
wait = WebDriverWait(driver, 20)

# === FUNCTION TO SCRAPE COMMENTARY ===
def scrape_commentary(url):
    try:
        driver.get(url)
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
            pass

        content_xpath = "/html/body/div[1]/main/div[2]/div[2]/div[1]/div[1]/div[7]/div[4]"
        content_div = driver.find_element(By.XPATH, content_xpath)
        lines = content_div.text.strip().split('\n')

        events = []
        i = 0
        while i < len(lines) - 1:
            if lines[i].endswith("'"):
                minute_text = lines[i].strip()
                desc = lines[i + 1].strip()
                try:
                    minute = int(minute_text.replace("'", "").replace("+", "").strip())
                    events.append((minute, desc))
                except:
                    pass
                i += 2
            else:
                i += 1

        events.reverse()

        def get_part_text(min_start, min_end=None):
            part = [desc for min, desc in events if min >= min_start and (min_end is None or min < min_end)]
            return " ".join(part).strip()

        return [
            get_part_text(0, 25),
            get_part_text(25, 45),
            get_part_text(45, 70),
            get_part_text(70)
        ]

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None
    
    
from openai import OpenAI

# Configure the OpenRouter API client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

def generate_report_from_parts(part1, part2, part3, part4):
    full_commentary = f"{part1}\n\n{part2}\n\n{part3}\n\n{part4}"
    
    completion = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "https://example.com",
            "X-Title": "Football Analysis Tool",
        },
        model="nvidia/llama-3.3-nemotron-super-49b-v1:free",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a football match analyst. Generate a clean, single-paragraph summary "
                    "of the match based only on the key events mentioned in the commentary. "
                    "Include goals, red cards, standout performances, and disciplinary notes "
                    "only if relevant. Do not include scores unless mentioned in the commentary. "
                    "Do not use phrases like 'Here is the report' or any buzzwords. Just return one tight paragraph."
                )
            },
            {
                "role": "user",
                "content": f"Generate a concise (about 100 words) match report from the following report, Fix any logical error etc :\n\n{full_commentary}"
            }
        ]
    )

    return completion.choices[0].message.content


# === LOAD MODEL & TOKENIZER ===
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("Model loaded.")

# === GENERATE REPORT ===
def generate_report(commentary_text):
    instruction = "Convert this football match commentary into a complete match report: "
    full_input = instruction + commentary_text
    inputs = tokenizer(full_input, return_tensors="pt", padding=True, truncation=True).to(DEVICE)

    with torch.no_grad():
        summary_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=300,
            num_beams=4,
            early_stopping=True
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# === MAIN FLOW ===
if __name__ == "__main__":
    url = input("Enter the match commentary URL: ")
    parts = scrape_commentary(url)

    if not parts:
        print("Failed to extract commentary.")
        exit()
    reports = []
    for i, part in enumerate(parts):
        print(f"\n===== PART {i+1} - INPUT COMMENTARY =====\n")
        print(part if part else "[No Commentary]")
        print(f"\n===== PART {i+1} - GENERATED REPORT =====\n")
        if part:
            result = generate_report(part)
            print(result)
            reports.append(result)
        else:
            print("[No Output]")
    print(generate_report_from_parts(reports[0],reports[1],reports[2],reports[3]))
