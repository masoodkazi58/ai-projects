import pickle
import streamlit as st 
import re 
import nltk 
from nltk.corpus import stopwords
import pdfplumber
import numpy as np

#loading resume_classifier model
clf = pickle.load(open('rf_model.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))

categories = {
    15:"Java Developer",
    20:"Python Developer",
    23:"Testing",
    8:"DevOps Engineer",
    24:"Web Designing",
    12:"HR",
    13:"Hadoop",
    3:"Blockchain",
    10:"ETL Developer",
    18:"Operations Manager",
    6:"Data Science",
    22:"Sales",
    16:"Mechanical Engineer",
    1:"Arts",
    7:"Database",
    11:"Electrical Engineering",
    14:"Health and fitness",
    19:"PMO",
    4:"Business Analyst",
    9:"DotNet Developer",
    2:"Automation Testing",
    17:"Network Security Engineer",
    21:"SAP Developer",
    5:"Civil Engineer",
    0:"Advocate",
}

def clean_text(txt):


    # 1. Remove URLs
    txt1 = re.sub(r'https?://\S+|www\.\S+', ' ', txt)
    # 2. Remove Gmail addresses
    txt2 = re.sub(r'[A-Za-z0-9._%+-]+@g+[A-Za-z0-9._%+-]+\.com', ' ', txt1)

    # 3. Remove @gmail.com text alone (if present)
    txt3 = re.sub(r'@gmail\.com', ' ', txt2)

    # 4. Remove hashtags (#) and mentions (@word)
    txt4 = re.sub(r'[@#]\w+', ' ', txt3)

    # 5. Remove all symbols/punctuation
    txt5 = re.sub(r'[^A-Za-z0-9\s]', ' ', txt4)

    # 6. Remove extra spaces
    txt6 = re.sub(r'\s+', ' ', txt5).strip()

    # 7. Remove stopwords
    stop_words = set(stopwords.words('english'))
    txt7 = " ".join([word for word in txt6.split() if word.lower() not in stop_words])

    return txt7

def pdf_txt(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


#website
def main():
    st.title("Resume Screen App")
    upload_file = st.file_uploader("upload Resume",type=['pdf'])
    if upload_file is not None:
        text_data = pdf_txt(upload_file)
        cleaned_resume = tfidf.transform([text_data])
        predicted = clf.predict(cleaned_resume)[0]
        category_name = categories.get(predicted,'Unlown')
        # st.write(category_name)
        # Predict probabilities
        probs = clf.predict_proba(cleaned_resume)[0]

        # Sort by highest probability
        sorted_idx = np.argsort(probs)[::-1]

        st.subheader("Best Match")
        st.success(f"{categories[predicted]} — {round(probs[predicted]*100, 2)}% fit")

        st.write("---")
        st.subheader("All Job Fit Percentages")
        
        i = 0
        for idx in sorted_idx:
            if i > 5:
                break
            job = categories[idx]
            pct = round(probs[idx] * 100, 2)
            st.write(f"**{job}: {pct}%**")
            i += 1
        


if __name__=="__main__":
    main()
