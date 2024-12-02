from flask import Flask, render_template, request
import pickle


app = Flask(__name__)

model_file = 'model.pkl'
vectorizer_file = 'tfidf_vectorizer.pkl'

with open(model_file, 'rb') as file:
    model = pickle.load(file)

with open(vectorizer_file, 'rb') as file:
    tfidf_vectorizer = pickle.load(file)


def predict_news(news):
    news_tfidf = tfidf_vectorizer.transform([news])
    prediction = model.predict(news_tfidf)
    return "REAL" if prediction[0] == 1 else "FAKE"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    prediction = predict_news(news)
    return render_template('index.html', prediction=prediction, news=news)

if __name__ == '__main__':
    app.run(debug=True)
