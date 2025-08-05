import numpy as np
import  pandas as pd
import nltk
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


True_news= pd.read_csv('True.csv')
False_news= pd.read_csv('Fake.csv')
True_news['label']=0
False_news['label']=1
dataset1=True_news[['text','label']]
dataset2=False_news[['text','label']]
dataset=pd.concat([dataset1,dataset2])
dataset = dataset.sample(frac=1)
ps = WordNetLemmatizer()
stopwords = stopwords.words('english')
nltk.download('wordnet')
def clean_row(row):
    row=row.lower()
    row=re.sub('[^a-zA-Z]', ' ', row)
    token = row.split()
    news= [ps.lemmatize(word) for word in token if not word in stopwords]
    cleaned_news = ' '.join(news)
    return cleaned_news
dataset['text']= dataset['text'].apply(lambda x: clean_row(x))
vectorizer = TfidfVectorizer(max_features=50000,lowercase=False,ngram_range=(1, 2))
X=dataset.iloc[:35000,0].values
y=dataset.iloc[:35000,1].values
train_data, test_data, train_label, test_label = train_test_split(X, y, test_size=0.2, random_state=0)
vec_train_data= vectorizer.fit_transform(train_data).toarray()
vec_test_data= vectorizer.transform(test_data).toarray()
train_data = pd.DataFrame(vec_train_data, columns=vectorizer.get_feature_names_out())
test_data = pd.DataFrame(vec_test_data, columns=vectorizer.get_feature_names_out())
clf= MultinomialNB()
clf.fit(train_data, train_label)
y_pred= clf.predict(test_data)
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(test_label, y_pred)*100)
print("Accuray on training data:", accuracy_score(train_label, clf.predict(train_data))*100)
txt=input("Enter a news headline to classify: ")
news=clean_row(str(txt))
pred = vectorizer.transform([txt]).toarray()

if clf.predict(pd.DataFrame(pred, columns=vectorizer.get_feature_names_out()))[0]==0:
    print("The news is True")
else:
    print("The news is False")
