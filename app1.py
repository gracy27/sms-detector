import streamlit as st
import pickle
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string

nltk.download('punkt')
nltk.download('stopwords')
ps = PorterStemmer()

def transform_text(text):
    text=text.lower() #LOWER CASE
    text=nltk.word_tokenize(text) # TOKENISATION
    # REMOVING SPECIAL CHARACTERS
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:] # cloning 
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
         y.append(i)
    
    text=y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


vectorizer_path = r"C:\Users\DELL\vectorizer.pkl"
print("Vectorizer path:", vectorizer_path)


with open(vectorizer_path, 'rb') as file:
    tfidf = pickle.load(file)


model_path=r"C:\Users\DELL\model.pkl"
print("Model path:", model_path)


with open(model_path, 'rb') as file:
    model = pickle.load(file)

st.title("SMS DETECTOR")
input_sns=st.text_input("ENTER THE MESSAGE:")
if st.button('PREDICT'):
#1.preprocess
       transform_sms=transform_text(input_sns)
#2 vectorize
       vector_input=tfidf.transform([transform_sms])
#3 predict
       result=model.predict(vector_input)[0]
#4 Display
       if(result==1):
        st.header("SPAM")
       else:
        st.header("NOT SPAM")