
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from customer_support_agent import DEMO_QUESTIONS

# 1. Convert questions into numerical features
vectorizer = TfidfVectorizer(stop_words="english")
features = vectorizer.fit_transform(DEMO_QUESTIONS)

# 2. Group similar questions into 6 clusters
model = KMeans(
    n_clusters=6,
    random_state=42,
    n_init=10
)

clusters = model.fit_predict(features)

# 3. Display the groups
for cluster_id in range(6):

    print(f"\nCluster {cluster_id + 1}")

    for question, cluster in zip(DEMO_QUESTIONS, clusters):
        if cluster == cluster_id:
            print("-", question)
