# My-First-classification-project-
I built a machine learning system that classifies messages as Spam or Not Spam using Naive Bayes and SVM.
import kagglehub
import pandas as pd
import os


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB 
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

#let's Load dataset from Kaggle


# Download last version
path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
print("Path to dataset files:", path)

# Check files in the directory (important for debugging)
#I used this line because the Previous line returns the folder, not file
print(os.listdir(path))

# Correct file name
#spam.Csv is name of the target file
#I used this line because the Previous line returns the folder, not file
csv_path = os.path.join(path, "spam.csv")

# Load dataset
data = pd.read_csv(csv_path, encoding="latin-1")[["v1", "v2" ]]
data.columns=["label", "text" ]

#convert labels
data["label"] =data['label'].map({'ham':0,'spam':1})
#traintest split
x_train,x_test,y_train,y_test=train_test_split(data["text"], data["label"], test_size=0.2,random_state=42)

#TF-IDF
#convert text to array
vectorizer=TfidfVectorizer(stop_words='english' )
x_train_tfidf=vectorizer.fit_transform(x_train)
x_test_tfidf=vectorizer.transform(x_test)

#Naive Bayes
nb=MultinomialNB() 
nb.fit(x_train_tfidf,y_train)
nb_pred=nb.predict(x_test_tfidf)

#SVM
svm=LinearSVC()
svm.fit(x_train_tfidf,y_train)
svm_pred=svm.predict(x_test_tfidf)

print('Naive Bayes Accuracy :', accuracy_score(y_test, nb_pred))

print ('SVM Accuracy :', accuracy_score(y_test, svm_pred) )
Email/SMS Spam Detection using Naive Bayes & SVM

I built a machine learning–based spam detection system that classifies messages as Spam or Not Spam using Natural Language Processing (NLP).
