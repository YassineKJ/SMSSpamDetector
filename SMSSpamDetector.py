import pandas as pd
import numpy as np
import re
import string
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

def strip_emoji(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese characters
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

#Remove punctuations, links, mentions and \r\n new line characters
def strip_all_entities(text):
    text = text.replace('\r', '').replace('\n', ' ').replace('\n', ' ').lower() #remove \n and \r and lowercase
    text = re.sub(r"(?:\@|https?\://)\S+", "", text) #remove links and mentions
    text = re.sub(r'[^\x00-\x7f]',r'', text) #remove non utf8/ascii characters such as '\x9a\x91\x97\x9a\x97'
    banned_list= string.punctuation + 'Ã'+'±'+'ã'+'¼'+'â'+'»'+'§'
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    return text

#clean hashtags at the end of the sentence, and keep those in the middle of the sentence by removing just the # symbol
def clean_hashtags(text):
    new_text = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', text)) #remove last hashtags
    new_text2 = " ".join(word.strip() for word in re.split('#|_', new_text)) #remove hashtags symbol from words in the middle of the sentence
    return new_text2

#Filter special characters such as & and $ present in some words
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

def to_lowercase(text):
    text = text.lower()
    return text

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

data = pd.read_csv(r'C:\Users\moha2\Downloads\Dataset_5971\Dataset_5971.csv', encoding='latin1')
# category_counts = data['LABEL'].value_counts()
# plt.figure(figsize=(6,4))
# category_counts.plot(kind='pie', autopct='%1.1f%%')
# plt.axis('equal')
# plt.title('Pie Chart of Distribution')
# plt.legend()
# plt.show()
y = data.LABEL
X = data.drop(['LABEL'], axis=1)
#data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
data = data.drop_duplicates()
print(data.head(5))
data.info()
data['LABEL'] = data['LABEL'].apply(to_lowercase)
data['URL'] = data['URL'].apply(to_lowercase)
data['EMAIL'] = data['EMAIL'].apply(to_lowercase)
data['PHONE'] = data['PHONE'].apply(to_lowercase)
data['TEXT'] = data['TEXT'].apply(strip_emoji).apply(strip_all_entities).apply(clean_hashtags).apply(filter_chars).apply(remove_mult_spaces)
data['TEXT'] = data['TEXT'].apply(remove_stopwords).apply(lemmatie_text)
data['text_length'] = data['TEXT'].apply(lambda x: len(str(x).split()))

data = data[data['text_length'] >= 2]
# Shuffle training dataframe
data = data.sample(frac=1, random_state=42) # shuffle with random_state=42 for reproducibility

le = LabelEncoder()
data['LABEL'] = le.fit_transform(data['LABEL'])
encodedData = {
    'Label': le.classes_,
    'Label Encoded': le.transform(le.classes_)
}
data['PHONE'] = le.fit_transform(data['PHONE'])
data['EMAIL'] = le.fit_transform(data["EMAIL"])
data['URL'] = le.fit_transform(data[['URL']])
encodedData = {
    'Label': le.classes_,
    'Label Encoded': le.transform(le.classes_)
}
dr = pd.DataFrame(encodedData)
print(dr)

ros = RandomOverSampler()
features = data[['TEXT', 'EMAIL', 'PHONE', 'URL']]
train_x, train_y = ros.fit_resample(features, data['LABEL'])
train_os = pd.DataFrame(train_x, columns=['TEXT', 'EMAIL', 'PHONE', 'URL'])
train_os['LABEL'] = train_y
train_os.head()
# Shuffle training dataframe
train_os = train_os.sample(frac=1, random_state=42) # shuffle with random_state=42 for reproducibility
y = train_os['LABEL']
X = train_os.drop(['LABEL'], axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X,y, train_size=0.8, test_size=0.2, random_state=0)
preprocessor = ColumnTransformer(
    transformers=[
        ('tfidf', TfidfVectorizer(), 'TEXT'),  # Apply TfidfVectorizer to the TEXT column
        ('passthrough', 'passthrough', ['EMAIL', 'PHONE', 'URL'])  # Pass the binary columns as-is
    ]
)
model_0 = Pipeline([
                    ("preprocessor", preprocessor), # convert words to numbers using tfidf
                    ("model", MultinomialNB()) # model the text
])
model_0.fit(X, y)
joblib.dump(model_0, 'sms_spam_detector.pkl')
#baseline_predicts = model_0.predict(X_valid)
#model_accuracy = accuracy_score(y_valid, baseline_predicts) * 100
#print(model_accuracy)

