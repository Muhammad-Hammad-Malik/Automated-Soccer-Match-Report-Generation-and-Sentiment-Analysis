import requests
import json

response = requests.get(
  url="https://openrouter.ai/api/v1/auth/key",
  headers={
    "Authorization": f"Bearer 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"
  }
)

print(json.dumps(response.json(), indent=2))
