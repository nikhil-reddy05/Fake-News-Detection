# Fake News Detection
---

## 🔍 Problem Statement

Fake news is a major issue in digital journalism and social media. The goal of this project is to **automatically classify news articles as "Fake" or "Real"** using Natural Language Processing (NLP) and Machine Learning/Deep Learning models.

---

## 📊 Dataset

- Dataset Source: [Kaggle – Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Total Rows: ~44,000 articles
- Columns: `title`, `text`, `subject`, `date`, `label`

---

## ⚙️ Features Implemented

### ✅ **Data Preprocessing & EDA**
- Merged and labeled `True.csv` and `Fake.csv`
- Cleaned HTML, URLs, punctuation, stopwords
- Analyzed text length, word count, NER entities
- Visualized class balance, word clouds, and NER distribution

### ✅ **Text Representation**
- TF-IDF Vectorization
- Word2Vec Embeddings
- GloVe 100d Pretrained Embeddings (for Deep Learning)

### ✅ **Machine Learning Models**
- Logistic Regression (with GridSearchCV)
- Naïve Bayes
- Support Vector Machine (SVM)
- Random Forest

### ✅ **Deep Learning Model**
- **Bidirectional LSTM** with:
  - Pretrained GloVe embeddings (100d)
  - Two stacked BiLSTM layers
  - Dropout & L2 regularization
  - EarlyStopping and ReduceLROnPlateau

### ✅ **Evaluation**
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Cross-validation (for classical models)

---

## 🧠 Model Performance Summary

| Model              | Accuracy |
|-------------------|----------|
| Logistic Regression | 98.7%    |
| Naïve Bayes         | 93.3%    |
| SVM                 | 99.4%    |
| Random Forest       | 99.8%    |
| **BiLSTM (GloVe)**  | **99.9%** ✅ |

> ⚠️ Note: Models trained on padded & tokenized text with proper validation strategy.

---

## 🧪 Tools & Technologies

- **Languages**: Python
- **Libraries**: NumPy, pandas, scikit-learn, TensorFlow/Keras, NLTK, Matplotlib, Seaborn
- **Embeddings**: GloVe (100d), Word2Vec
- **EDA Tools**: SpaCy, WordCloud
- **Version Control**: Git, GitHub

---

## 🚀 Deployment (Optional)
Can be deployed via:
- Flask API for local predictions
- Streamlit web app for UI

---

## 💡 Future Improvements
- Integrate BERT or DistilBERT for transformer-based classification
- Add SHAP/LIME for explainability
- Self-learning (active learning loop)
- Streamlit UI for production-level deployment
- Deploy on Render / HuggingFace Spaces / GCP

---

## 🤝 Acknowledgements
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- [Kaggle Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Inspired by the need for factual integrity in online journalism.

---

## 👨‍💻 Author
**Nikhil Reddy Banda**  
[GitHub](https://github.com/nikhil-reddy05) | [LinkedIn](https://linkedin.com/in/nikhil-reddy05)  
