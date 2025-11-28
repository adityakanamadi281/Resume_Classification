import pandas as pd
import pickle 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
import joblib 
import os




PROCESSED_PATH = Path(r"C:\Users\adity\Resume_Classification\Resumes\Cleaned_Resumes.csv")

df = pd.read_csv(PROCESSED_PATH)

# Features
x = df['Resume_Details'].values
y = df['Category'].values

x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2, random_state=42)

tfidf_vector = TfidfVectorizer(sublinear_tf=True,stop_words='english')
x_train = tfidf_vector.fit_transform(x_train)
x_test = tfidf_vector.transform(x_test)


# Models 
model_DT = DecisionTreeClassifier(criterion='gini')
model_DT.fit(x_train, y_train)


model_RF = RandomForestClassifier(n_estimators=200)
model_RF.fit(x_train, y_train)

model_svm = SVC()
model_svm.fit(x_train, y_train)



save_path = r"C:\Users\adity\Resume_Classification\models"
os.makedirs(save_path, exist_ok=True)
filename = os.path.join(save_path, "model_RF.pkl")
pickle.dump(model_RF, open(filename, 'wb'))



vectorizer_filename = os.path.join(save_path, "VECTOR.pkl")
pickle.dump(tfidf_vector, open(vectorizer_filename, 'wb'))

