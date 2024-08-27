from flask import Flask, request, jsonify
import joblib
import re
import string
import pandas as pd
import numpy as np
from flask_cors import CORS
import nltk
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
CORS(app)

model = joblib.load('sms_spam_detector.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    contains_url = 0
    contains_email = 0
    data = request.json
    message = data['message']
    message = preprocess(message)
    if message.find("http") != -1:
        contains_url = 1
    contains_phone = contains_phone_number(message)
    if re.search(r'[\w.]+\@[\w.]+', message) != None:
        contains_email = 1
    input_data = pd.DataFrame([[message, contains_url, contains_email, contains_phone]], columns=['TEXT','URL','EMAIL','PHONE'])
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        return jsonify({'prediction': "Spam!"})
    return jsonify({'prediction': "Not Spam"})
    
def preprocess(message):
    message = clean_hashtags(strip_all_entities(message))
    message = remove_mult_spaces(filter_chars(message))
    message = lemmatie_text(remove_stopwords(message))
    return message

def contains_phone_number(message):
    extract_phone_number_pattern = "\\+?[1-9][0-9]{7,14}"
    list_of_phone_numbers = re.findall(extract_phone_number_pattern, message)
    if len(list_of_phone_numbers) > 0:
        return 1
    return 0

def strip_all_entities(text):
    text = text.replace('\r', '').replace('\n', ' ').replace('\n', ' ').lower() #remove \n and \r and lowercase
    text = re.sub(r"(?:\@|https?\://)\S+", "", text) #remove links and mentions
    text = re.sub(r'[^\x00-\x7f]',r'', text) #remove non utf8/ascii characters such as '\x9a\x91\x97\x9a\x97'
    banned_list= string.punctuation + 'Ã'+'±'+'ã'+'¼'+'â'+'»'+'§'
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    return text

def clean_hashtags(text):
    new_text = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', text)) #remove last hashtags
    new_text2 = " ".join(word.strip() for word in re.split('#|_', new_text)) #remove hashtags symbol from words in the middle of the sentence
    return new_text2

def filter_chars(a):
    sent = []
    for word in a.split(' '):
        if ('$' in word) | ('&' in word):
            sent.append('')
        else:
            sent.append(word)
    return ' '.join(sent)

def remove_mult_spaces(text): # remove multiple spaces
    return re.sub("\s\s+" , " ", text)

def remove_stopwords(text):
    stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]

    # Sentence converted to lowercase-only
    text = text.lower()

    words = text.split()
    non_stopwords = [w for w in words if w not in stopwords]
    text_without_stopwords = " ".join(non_stopwords)

    return text_without_stopwords

lemmatizer = WordNetLemmatizer()
def lemmatie_text(text):
    word_list = nltk.word_tokenize(text)
    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    return lemmatized_output

if __name__ == '__main__':
    app.run(debug=True)