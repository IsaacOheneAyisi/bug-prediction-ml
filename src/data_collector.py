import requests
import pandas as pd

url = "https://api.github.com/repos/tensorflow/tensorflow/commits"

response = requests.get(url)
data = response.json()

messages = []

for commit in data[:50]:
    messages.append(commit["commit"]["message"])

df = pd.DataFrame(messages, columns=["message"])

df.to_csv("data/commits.csv", index=False)

print("Data saved to data/commits.csv")