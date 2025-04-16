import pandas as pd
from openai import OpenAI

# Configure the OpenRouter API client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

# Function to generate match report using the deepseekr1 model
def generate_match_report(commentary):
    completion = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "https://example.com",
            "X-Title": "Football Analysis Tool",
        },
        model="google/gemini-2.0-flash-lite-001",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a football match analyst. Generate a clean, single-paragraph summary "
                    "based only on the key events mentioned in this *part* of a match commentary. "
                    "Include goals, red cards, standout performances, or discipline if relevant. "
                    "Don't assume score unless mentioned. Avoid phrases like 'Here is a report'. "
                    "Just return one paragraph (around 40 to 50 words)."
                )
            },
            {
                "role": "user",
                "content": f"{commentary.strip()}"
            }
        ]
    )
    return completion.choices[0].message.content.strip()

# Main process
def process_sequentially(input_csv, output_csv, start_index=0):
    try:
        df = pd.read_csv(input_csv)

        if 'Commentary' not in df.columns:
            raise ValueError("CSV must contain a 'Commentary' column")

        if 'MatchReport' not in df.columns:
            df['MatchReport'] = ""

        total = len(df)

        for idx in range(start_index, total):
            commentary = str(df.at[idx, 'Commentary']).strip()

            if not commentary or commentary.lower() == 'nan':
                print(f"⚠️  Skipping empty commentary at row {idx}")
                continue

            print(f"\n▶ Processing row #{idx} — {len(commentary.split())} words")

            try:
                report = generate_match_report(commentary)
                df.at[idx, 'MatchReport'] = report
                df.to_csv(output_csv, index=False)

                remaining = total - (idx + 1)
                print(f" Done. {remaining} rows remaining.")

            except Exception as e:
                print(f"\n Error at index {idx}: {e}")
                print(" Halting process. You can resume from this index.")
                break

        print(f"\n Finished. Output saved to {output_csv}")

    except Exception as e:
        print(f"\n Fatal error: {e}")

# Run script
if __name__ == "__main__":
    input_csv = "commentary_dataset.csv"  # Matches your earlier CSV name
    output_csv = "commentary_dataset_with_reports.csv"
    process_sequentially(input_csv, output_csv, start_index=0)
