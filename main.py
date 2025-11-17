#first : data_cleaning.py
#second : visualise_data.ipynb
#third : data_preprocessing.py
#four : train_model.py to get best model to train 


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from data_preprocessing import transform_text

MODEL_FILE = "model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"

while True:
    if not os.path.exists(MODEL_FILE):
        tfid = TfidfVectorizer(max_features=3000)

        df = pd.read_csv("CSV_Files/spam_final.csv")
        df = df.dropna(subset=['transformed_text'])
        df = df[df['transformed_text'].str.strip() != '']
        df = df.reset_index(drop=True)

        X = tfid.fit_transform(df['transformed_text']).toarray()
        y = df['target'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

        clf = ExtraTreesClassifier(n_estimators=50, random_state=42)

        print("Training.......")
        clf.fit(X_train, y_train)
        joblib.dump(clf, MODEL_FILE)
        joblib.dump(tfid, VECTORIZER_FILE)
        print("Model and vectorizer saved.")

    else:
        model = joblib.load(MODEL_FILE)
        tfid = joblib.load(VECTORIZER_FILE)

        new_mail = input("Enter Mail: ")
        processed_mail = transform_text(new_mail)
        X = tfid.transform([processed_mail]).toarray()

        y_pred = model.predict(X)
        print("Prediction:", "Spam" if y_pred[0] == 1 else "Not Spam")

        Exit = input("Continue or Not??: ").strip().lower()
        if Exit in ["not", "no", "n", "exit", "stop"]:
            break
