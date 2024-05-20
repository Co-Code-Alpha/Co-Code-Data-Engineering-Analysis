import requests

url = "https://6832-34-125-51-159.ngrok-free.app/generate"
data = {
    "instruction": "곽우진은 롤 다이아를 찍을수있을까?"
}

response = requests.post(url, json=data)
print(response.json())