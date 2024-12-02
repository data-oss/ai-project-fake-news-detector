import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

fake_data = pd.read_csv('Fake.csv')
true_data = pd.read_csv('True.csv')
fake_data['label'] = 0
true_data['label'] = 1
data = pd.concat([fake_data, true_data], ignore_index=True)
a = data['text']
b = data['label']
a_train, a_test, b_train, b_test = train_test_split(a,b,test_size=0.2,random_state=42)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
a_train_tfidf = tfidf_vectorizer.fit_transform(a_train)
a_test_tfidf = tfidf_vectorizer.transform(a_test)
b_train_tfidf = tfidf_vectorizer.transform(b_train)
b_test_tfidf = tfidf_vectorizer.transform(b_test)
model = LogisticRegression()
model.fit(a_train_tfidf, b_train)

with open('model.pkl','wb') as model_file:
    pickle.dump(model, model_file)

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)
print("Model and vectorizer saved successfully!")
