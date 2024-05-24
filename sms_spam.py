import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import joblib

# 데이터 로드
# 데이터 형태는 Message 내용인 message Column과 Spam인지 아닌지를 나타내는 label Column으로 구성되어 있음
# Spam 문자는 spam, 일반 문자는 ham으로 구분
data = pd.read_csv('/home/moonguigon/work/sms_spam/sms_spam.csv')
X = data['message']
y = data['label']

# 데이터 전처리 및 벡터화
# tf-idf 결합을 통해 단어의 가중치를 나타냄(TfidfVectorizer())
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# 모델 학습
model = MultinomialNB()
model.fit(X_train, y_train)