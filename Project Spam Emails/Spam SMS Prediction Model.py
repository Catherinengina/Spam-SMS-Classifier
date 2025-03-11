#!/usr/bin/env python
# coding: utf-8

# Import Libraries

# In[27]:


import pandas as pd
import numpy as np
import re   
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt 

nltk.download('stopwords')


# Load the dataset & clean

# In[28]:


df = pd.read_csv('Project Spam Emails/spam.csv', encoding='latin-1')
print(df.head())


# In[29]:


print(df.tail())


# In[30]:


df = df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"],errors="ignore")


# In[31]:


print(df.head())


# In[32]:


print(df.columns)


# In[33]:


df = df.rename(columns={"v1": "label", "v2": "message"}) 


# In[34]:


print(df.head())


# In[35]:


print(df.info())


# In[36]:


df['label'] = df['label'].map({'ham': 0, 'spam': 1})


# In[37]:


print(df.head())


# In[40]:


def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.split()
    text = [word for word in text if word not in stopwords.words('english')]  # Remove stopwords
    return " ".join(text)


# In[41]:


df['cleaned_message'] = df['message'].apply(clean_text) 


# Train-Test Split

# In[54]:


X_train, X_test, y_train, y_test = train_test_split(df['cleaned_message'], df['label'], test_size=0.3, random_state=4000)  


# Convert Text to Numerical format (TF-IDF features)

# In[55]:


vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


#  Naïve Bayes Model

# In[56]:


model = MultinomialNB()
model.fit(X_train_tfidf, y_train) 


# Prediction

# In[57]:


y_pred = model.predict(X_test_tfidf) 


# Evaluate accuracy

# In[58]:


accuracy = accuracy_score(y_test, y_pred) 
print(f"Accuracy: {accuracy:.2f}") 
print(classification_report(y_test, y_pred))


# Test with Sample SMS

# In[59]:


def predict_spam(text):
    cleaned_text = clean_text(text)
    text_tfidf = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_tfidf)[0]
    return "Spam" if prediction == 1 else "Ham"


# In[60]:


sample_sms = "URGENT! Your bank account has been compromised. Log in immediately to verify your details!"
print(f"Sample SMS: {sample_sms}")
print(f"Prediction: {predict_spam(sample_sms)}")


# In[61]:


sample_sms = "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight"
print(f"Sample SMS: {sample_sms}")
print(f"Prediction: {predict_spam(sample_sms)}")


# In[62]:


sample_sms = "URGENT! You have won a 1 week FREE membership in our �100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18"
print(f"Sample SMS: {sample_sms}")
print(f"Prediction: {predict_spam(sample_sms)}")


# In[66]:


new_text = ["Free entry to win $1000! click here now!"]
cleaned_new_text = [clean_text(text) for text in new_text]
new_text_vectorized = vectorizer.transform(cleaned_new_text)
prediction = model.predict(new_text_vectorized)
print(f"Prediction for new text: {prediction}")


# In[67]:


new_text = ["What time are we meeting for lunch tomorrow?"]
cleaned_new_text = [clean_text(text) for text in new_text]
new_text_vectorized = vectorizer.transform(cleaned_new_text)
prediction = model.predict(new_text_vectorized)
print(f"Prediction for new text: {prediction}")

# Save the model and vectorizer using pickle
try:
    # Save the model
    with open('spam_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save the TF-IDF vectorizer
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("Model and vectorizer trained and saved.")
except Exception as e:
    print(f"Error during training: {e}")

# Load the model and vectorizer for prediction using pickle
try:
    with open('spam_model.pkl', 'rb') as f:  # Changed file extension and loading
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:  # Changed file extension and loading
        vectorizer = pickle.load(f)
    print("Model and vectorizer loaded successfully.")
except Exception as e:
    print(f"Error loading model/vectorizer: {e}")

# Prediction function
def predict_spam(text):
    cleaned_text = clean_text(text)
    text_tfidf = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_tfidf)[0]
    return "Spam" if prediction == 1 else "Ham"

# Streamlit UI
st.title("Spam SMS Prediction")
user_input = st.text_area("Enter your SMS text here:")

if st.button("Predict"):
    if user_input:
        try:
            prediction = predict_spam(user_input)
            st.write(f"Prediction: {prediction}")
        except Exception as e:
            st.error(f"Prediction error: {e}")
            print(f"Prediction error: {e}")
    else:
        st.warning("Please enter some text.")



