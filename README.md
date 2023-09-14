# Task_5
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Loading the dataset
df = pd.read_csv("https://catalog.data.gov/dataset/consumer-complaint-database")

# Data preprocessing and feature engineering
# Handling missing values, clean data, and extract features
# Spliting the dataset into training and testing sets
X = df['complaint_text']  # Replace 'complaint_text' with your text column name
y = df['category']  # Replace 'category' with your label column name
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Text vectorization using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Adjust max_features as needed
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Training a multi-class classification model (e.g., Multinomial Naive Bayes)
model = OneVsRestClassifier(MultinomialNB())
model.fit(X_train_tfidf, y_train)

# Making predictions
y_pred = model.predict(X_test_tfidf)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", report)
