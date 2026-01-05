import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# 1. Load Data
df = pd.read_csv('fake_news.csv') 
# Assume labels are: 'FAKE' or 'REAL'

# 2. Split Data (Training vs Testing)
x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)

# 3. Vectorization (Convert Text -> Numbers)
# 'stop_words' removes common words like 'and', 'the'
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = vectorizer.fit_transform(x_train) 
tfidf_test = vectorizer.transform(x_test)

# 4. Train Model
# PassiveAggressiveClassifier is great for text data
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# 5. Predict & Evaluate
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {score*100:.2f}%')
