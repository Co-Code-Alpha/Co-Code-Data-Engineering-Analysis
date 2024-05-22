import requests

url = "https://256e-34-124-203-232.ngrok-free.app/generate"
data = {
    "instruction": "graph 자료구조 C++ 예시 코드 짜줘"
}

response = requests.post(url, json=data)
print(response.json())