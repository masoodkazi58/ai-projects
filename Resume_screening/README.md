# 🧠 Resume Screening & Job Role Prediction System

An end-to-end **machine learning–based resume screening application** that analyzes resumes and predicts the most suitable job roles using NLP and a trained classification model. The system provides **job fit percentages** across multiple roles through an interactive web interface.

---

## 🚀 Features
- Upload resumes in **PDF format**
- Extracts resume text automatically
- Performs **text cleaning and preprocessing**
- Converts text into numerical features using **TF-IDF**
- Predicts the **best-matched job role**
- Displays **fit percentages for multiple job categories**
- Interactive **Streamlit web application**

---

## 🧩 Project Structure
Resume_screening/
│
├── resume_web.py # Streamlit web application (inference + UI)
├── resume_screening.ipynb # Model training and experimentation
├── rf_model.pkl # Trained Random Forest model
├── tfidf.pkl # Trained TF-IDF vectorizer
├── clf.pkl # Serialized classifier (optional)
└── README.md

---

## 🛠️ Tech Stack
- **Python**
- **Machine Learning:** Scikit-learn (Random Forest)
- **NLP:** TF-IDF, NLTK
- **PDF Processing:** pdfplumber
- **Web Framework:** Streamlit
- **Utilities:** NumPy, Regex

---

## 🧪 Machine Learning Pipeline
1. **Resume Upload (PDF)**
2. **Text Extraction** using pdfplumber
3. **Text Cleaning**
   - URL & email removal  
   - Stopword removal  
   - Noise and symbol filtering  
4. **Feature Engineering**
   - TF-IDF vectorization  
5. **Model Inference**
   - Random Forest classifier predicts job category
   - Probability scores for each role

---

## 🖥️ How to Run the Application

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/masoodkazi58/ai-projects.git
cd ai-projects/Resume_screening
pip install streamlit nltk pdfplumber scikit-learn numpy
import nltk
nltk.download('stopwords')
streamlit run resume_web.py
