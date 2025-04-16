import pandas as pd
from openai import OpenAI

# Configure the OpenRouter API client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",  # Replace with your actual key
)

# Function to generate match report using the deepseekr1 model
def generate_match_report(commentary):
    completion = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "https://example.com",  # Optional. Site URL for rankings on openrouter.ai.
            "X-Title": "Football Analysis Tool",  # Optional. Site title for rankings on openrouter.ai.
        },
        model="google/gemini-2.0-flash-lite-001",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a football match analyst. Generate a clean, single-paragraph summary "
                    "of the match based only on the key events mentioned in the commentary. "
                    "Include goals, red cards, standout performances, and disciplinary notes "
                    "only if relevant. Do not include scores unless mentioned in the commentary. "
                    "Do not add any introductions like 'Here is a report'. Just return the paragraph."
                )
            },
            {
                "role": "user",
                "content": f"Generate a concise (about 100 words) match report from the following commentary:\n\n{commentary}"
            }
        ]
    )
    return completion.choices[0].message.content

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
            commentary = df.at[idx, 'Commentary']
            print(f"\n▶ Processing row #{idx} — {len(commentary.split())} words")

            try:
                report = generate_match_report(commentary)
                df.at[idx, 'MatchReport'] = report
                df.to_csv(output_csv, index=False)

                remaining = total - (idx + 1)
                print(f"✅ Done. {remaining} entries remaining.")

            except Exception as e:
                print(f"\n❌ Error at index {idx}: {e}")
                print("Halting process. You can resume from this index.")
                break

        print(f"\n🎉 Finished. Output saved to {output_csv}")

    except Exception as e:
        print(f"\n🔥 Fatal error: {e}")

# Run script
if __name__ == "__main__":
    input_csv = "Commentaries.csv"
    output_csv = "Commentaries_with_Reports.csv"
    process_sequentially(input_csv, output_csv, start_index=42)  # Change start_index as needed
