import streamlit as st
import pickle
import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

ps=PorterStemmer()
STOPWORDS = set(stopwords.words('english'))
tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in STOPWORDS and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)

st.title("SMS Spam Classifier")

input_sms=st.text_input("Enter the message")

if st.button("Predict"):
    #1 preprocessing
    transformed_sms=transform_text(input_sms)
    #2 vectorize
    vectorized_sms=tfidf.transform([transformed_sms]).toarray()
    #3 predict 
    prediction=model.predict(vectorized_sms)[0]
    #4 display
    if prediction == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")