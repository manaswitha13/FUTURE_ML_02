import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from preprocess import preprocess
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, "data", "customer_support_tickets.csv")
print("📂 Reading file from:", file_path)
df = pd.read_csv(file_path, encoding='latin1')
print("\n🔍 First 5 rows:")
print(df.head())
print("\n📊 Columns in dataset:")
print(df.columns)
df = df.rename(columns={
    "Ticket Description": "text",
    "Ticket Type": "category",
    "Ticket Priority": "priority"
})

# =========================
# STEP 4: SELECT REQUIRED COLUMNS
# =========================
df = df[['text', 'category', 'priority']].dropna()

print("\n✅ Cleaned Data Shape:", df.shape)

# =========================
# STEP 5: PREPROCESS TEXT
# =========================
df['clean_text'] = df['text'].apply(preprocess)

# =========================
# STEP 6: SPLIT DATA
# =========================
X = df['clean_text']
y_cat = df['category']
y_pri = df['priority']

X_train, X_test, y_cat_train, y_cat_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42
)

_, _, y_pri_train, y_pri_test = train_test_split(
    X, y_pri, test_size=0.2, random_state=42
)

# =========================
# STEP 7: CATEGORY MODEL
# =========================
cat_model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('clf', LogisticRegression(max_iter=200))
])

cat_model.fit(X_train, y_cat_train)

# =========================
# STEP 8: PRIORITY MODEL
# =========================
pri_model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression(max_iter=200))
])

pri_model.fit(X_train, y_pri_train)

# =========================
# STEP 9: EVALUATION
# =========================
print("\n📊 Category Model Performance:\n")
print(classification_report(y_cat_test, cat_model.predict(X_test)))

print("\n📊 Priority Model Performance:\n")
print(classification_report(y_pri_test, pri_model.predict(X_test)))

# =========================
# STEP 10: SAVE MODELS
# =========================
with open(os.path.join(BASE_DIR, "model_cat.pkl"), "wb") as f:
    pickle.dump(cat_model, f)

with open(os.path.join(BASE_DIR, "model_pri.pkl"), "wb") as f:
    pickle.dump(pri_model, f)

print("\n✅ Models saved successfully!")