import pandas as pd

df = pd.read_csv("data/commits.csv")

def label_commit(msg):
    keywords = ["fix", "bug", "error", "issue"]
    return 1 if any(k in msg.lower() for k in keywords) else 0

df["label"] = df["message"].apply(label_commit)

df.to_csv("data/commits_labeled.csv", index=False)

print("Labeled dataset created")