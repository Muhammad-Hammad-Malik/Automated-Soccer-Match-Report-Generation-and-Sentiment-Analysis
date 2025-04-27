from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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
from openai import OpenAI
from typing import List, Optional

# === SUPPRESS WARNINGS ===
warnings.filterwarnings("ignore")

# Optionally silence stderr
import sys
class DevNull:
    def write(self, msg): pass
    def flush(self): pass

sys.stderr = DevNull()

# === CONFIGURATION ===
CHROMEDRIVER_PATH = "chromedriver.exe"
MODEL_PATH = "match_report_model.pt"
DEVICE = torch.device("cpu")
MODEL_NAME = "google/flan-t5-small"

# === PYDANTIC MODELS ===
class MatchURLRequest(BaseModel):
    url: str

class MatchReport(BaseModel):
    quarter1_report: str
    quarter2_report: str
    quarter3_report: str
    quarter4_report: str
    overall_report: str

# === GLOBAL VARIABLES TO STORE MODELS ===
tokenizer = None
model = None
driver = None
wait = None
openai_client = None

# === INITIALIZE SELENIUM ===
def init_selenium():
    global driver, wait
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    service = Service(executable_path=CHROMEDRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=options)
    wait = WebDriverWait(driver, 20)

# === INITIALIZE OPENAI CLIENT ===
def init_openai_client():
    global openai_client
    openai_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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",
    )

# === INITIALIZE MODEL & TOKENIZER ===
def init_model_and_tokenizer():
    global tokenizer, model
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully.")

# === SCRAPE COMMENTARY ===
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
        raise HTTPException(status_code=500, detail=f"Failed to scrape commentary: {str(e)}")

# === GENERATE REPORT FROM PARTS ===
def generate_report_from_parts(part1, part2, part3, part4):
    full_commentary = f"{part1}\n\n{part2}\n\n{part3}\n\n{part4}"
    
    completion = openai_client.chat.completions.create(
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
                    "Do not use phrases like 'Here is the report' or any introduction phrases at all. "
                    "Start directly with the match details. No preamble or conclusion statements. "
                    "Just return one tight paragraph about the match events."
                )
            },
            {
                "role": "user",
                "content": f"Write a concise (about 100 words) match report from this commentary. Start immediately with the match details with no introductory phrases or meta-commentary:\n\n{full_commentary}"
            }
        ]
    )

    return completion.choices[0].message.content

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

# === FASTAPI APP ===
app = FastAPI(title="Football Match Report API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load models at startup
@app.on_event("startup")
async def startup_event():
    init_selenium()
    init_model_and_tokenizer()
    init_openai_client()
    print("API is ready to use!")

# Clean up resources on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    if driver:
        driver.quit()

@app.post("/generate-report", response_model=MatchReport)
async def generate_match_report(request: MatchURLRequest):
    try:
        # Scrape commentary
        parts = scrape_commentary(request.url)
        
        if not parts:
            raise HTTPException(status_code=404, detail="No commentary found at the provided URL")
        
        # Generate reports for each quarter
        quarter_reports = []
        for i, part in enumerate(parts):
            if part:
                report = generate_report(part)
                quarter_reports.append(report)
            else:
                quarter_reports.append("No commentary available for this part of the match.")
        
        # Generate overall report
        overall_report = generate_report_from_parts(
            quarter_reports[0], 
            quarter_reports[1], 
            quarter_reports[2], 
            quarter_reports[3]
        )
        
        return MatchReport(
            quarter1_report=quarter_reports[0],
            quarter2_report=quarter_reports[1],
            quarter3_report=quarter_reports[2],
            quarter4_report=quarter_reports[3],
            overall_report=overall_report
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}

# For running the app directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)