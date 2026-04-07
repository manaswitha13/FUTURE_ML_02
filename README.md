# 🎫 Support Ticket Classification & Prioritization System

## 📌 Overview
This project is a Machine Learning-based system that automatically classifies customer support tickets and assigns priority levels. It helps support teams handle tickets more efficiently by reducing manual effort and ensuring urgent issues are addressed quickly.

---

## 🚀 Features

- ✔ Text preprocessing using NLP (NLTK + spaCy)
- ✔ Automatic ticket classification (category prediction)
- ✔ Priority prediction (Critical / High / Medium / Low)
- ✔ Hybrid approach (Machine Learning + Rule-based logic)
- ✔ Interactive web app using Streamlit

---

## 🧠 How It Works

1. **Input**: User enters a support ticket (text)
2. **Preprocessing**:
   - Lowercasing
   - Removing special characters
   - Stopword removal
   - Lemmatization
3. **Feature Extraction**:
   - TF-IDF Vectorization (n-grams)
4. **Prediction**:
   - Category classification using ML model
   - Priority prediction using ML + rule-based logic
5. **Output**:
   - Displays category and priority

---

## 🛠️ Tech Stack

- Python
- Scikit-learn
- NLTK
- spaCy
- Streamlit

---

## 📊 Dataset

- Source: Kaggle Customer Support Ticket Dataset  
- Contains:
  - Ticket Description
  - Ticket Type (Category)
  - Ticket Priority

---

## 📁 Project Structure
