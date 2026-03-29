import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/commits_labeled.csv")

texts = df["message"].tolist()
labels = df["label"].tolist()

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(texts)

X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.3
)

clf = LogisticRegression()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print("LLM Model Accuracy:", accuracy)