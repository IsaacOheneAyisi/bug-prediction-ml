import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/commits_labeled.csv")

counts = df["label"].value_counts()

labels = ["Non-Bug", "Bug"]
values = [counts[0], counts[1]]

plt.bar(labels, values)
plt.title("Bug vs Non-Bug Commits")
plt.xlabel("Commit Type")
plt.ylabel("Count")

plt.savefig("results/plots.png")
plt.show()