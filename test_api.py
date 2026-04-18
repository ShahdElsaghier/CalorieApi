import requests

url = "http://127.0.0.1:5000/api/predict"

files = {
    "img": open("Burger.png", "rb")  
}

response = requests.post(url, files=files)

print("Status Code:", response.status_code)
print("Response:", response.json())