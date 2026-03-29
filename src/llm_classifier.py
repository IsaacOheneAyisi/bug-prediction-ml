from transformers import pipeline

classifier = pipeline("text-classification")

texts = [
    "fixed bug in login module",
    "added new feature",
]

results = classifier(texts)

for text, result in zip(texts, results):
    print(text, "->", result)