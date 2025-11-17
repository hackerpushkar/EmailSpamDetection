# Import all the required libraries (tools) that help us process data and train models
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # To convert text into numbers
import pandas as pd  # To handle and process data (like Excel or CSV files)
from sklearn.model_selection import train_test_split  # To split data for training and testing
from sklearn.linear_model import LogisticRegression  # One type of ML model
from sklearn.svm import SVC  # Support Vector Classifier model
from sklearn.naive_bayes import MultinomialNB  # Naive Bayes model
from sklearn.tree import DecisionTreeClassifier  # Decision Tree model
from sklearn.neighbors import KNeighborsClassifier  # K-Nearest Neighbors model
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier  # Different types of ensemble models (combined models)
from xgboost import XGBClassifier  # Another advanced model called XGBoost
from sklearn.metrics import accuracy_score, precision_score  # To check how good our models perform


# Create objects to convert text into numbers (because machines understand numbers, not text)
cv = CountVectorizer()  
tfid = TfidfVectorizer(max_features = 3000)  # Only keep top 3000 important words


# Read the data file (contains messages and their labels like spam or not spam)
df = pd.read_csv("CSV_Files/spam_final.csv")


# Remove any rows where the message text is missing or empty
df = df.dropna(subset=['transformed_text'])
df = df[df['transformed_text'].str.strip() != '']

# Reset the index numbers after removing empty rows
df = df.reset_index(drop=True)


# Convert the message text into numerical form using TF-IDF
X = tfid.fit_transform(df['transformed_text']).toarray()  # All messages become numeric arrays
y = df['target'].values  # 'target' column has 0 or 1 → (not spam or spam)


# Split the data into training and testing parts
# 80% data for training, 20% for testing
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)


# Create different machine learning models
svc = SVC(kernel= "sigmoid", gamma = 1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth = 5)
lrc = LogisticRegression(solver = 'liblinear', penalty = 'l1')
rfc = RandomForestClassifier(n_estimators = 50, random_state = 42)
abc = AdaBoostClassifier(n_estimators = 50, random_state = 42)
bc = BaggingClassifier(n_estimators = 50, random_state = 42)
etc = ExtraTreesClassifier(n_estimators = 50, random_state = 42)
gbdt = GradientBoostingClassifier(n_estimators = 50, random_state = 42)    
xgb  = XGBClassifier(n_estimators = 50, random_state = 42)


# Store all models in one dictionary for easy looping
clfs = {
    'SVC': svc,
    'KNN': knc,
    'NB': mnb,
    'DT': dtc,
    'LR': lrc,
    'RF': rfc,
    'Adaboost': abc,
    'Bgc': bc,
    'ETC': etc,
    'GBDT': gbdt,
    'xgb': xgb
}


# Function to train and test a model
def train_classifier(clfs, X_train, y_train, X_test, y_test):
    clfs.fit(X_train, y_train)  # Train model on training data
    y_pred = clfs.predict(X_test)  # Predict results on test data
    accuracy = accuracy_score(y_test, y_pred)  # Check how many predictions are correct
    precision = precision_score(y_test, y_pred)  # Check how accurate the positive predictions are
    return accuracy, precision


# Lists to store the accuracy and precision of all models
accuracy_scores = []
precision_scores = []

# Loop through all models and test them one by one
best_model_name = None
best_accuracy = 0
best_precision = 0

for name, clfs in clfs.items():
    current_accuracy, current_precision = train_classifier(clfs, X_train, y_train, X_test, y_test)
    
    print()
    print("For:", name)
    print("Accuracy:", current_accuracy)
    print("Precision:", current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)
    
    # Check if current model is better than the previous best
    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        best_precision = current_precision
        best_model_name = name

print("\n--------------------------------------------")
print("Best Model:", best_model_name)
print("Best Accuracy:", best_accuracy)
print("Best Precision:", best_precision)
print("--------------------------------------------")

