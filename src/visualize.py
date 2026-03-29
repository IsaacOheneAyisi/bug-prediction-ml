import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/commits_labeled.csv")

counts = df["label"].value_counts()

plt.bar(["Non-Bug", "Bug"], counts)
plt.title("Bug vs Non-Bug Commits")

plt.savefig("results/plots.png")