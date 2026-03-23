import requests
import json
import csv

CLIENT_ID = "CautdjkcAQPtttge1a0q"
CLIENT_SECRET = "70zAFqxJWf"

url = "https://openapi.naver.com/v1/datalab/search"

headers = {
    "X-Naver-Client-Id": CLIENT_ID,
    "X-Naver-Client-Secret": CLIENT_SECRET,
    "Content-Type": "application/json"
}

body = {
    "startDate": "2025-03-01",
    "endDate": "2026-03-22",
    "timeUnit": "date",
    "keywordGroups": [
        {
            "groupName": "꼬북칩",
            "keywords": ["꼬북칩"]
        }
    ]
}

response = requests.post(
    url,
    headers=headers,
    data=json.dumps(body, ensure_ascii=False).encode("utf-8")
)

data = response.json()

# CSV 저장
with open("naver_trend.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["date", "keyword", "ratio"])

    for result in data["results"]:
        keyword = result["title"]
        for d in result["data"]:
            writer.writerow([d["period"], keyword, d["ratio"]])

print("CSV 저장 완료")
