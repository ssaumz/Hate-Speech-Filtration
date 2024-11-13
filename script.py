import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Download necessary resources
nltk.download('punkt')
nltk.download('wordnet')

# Initialize FastAPI
app = FastAPI()

# Set up templates (if needed for serving HTML)
templates = Jinja2Templates(directory=".")

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Text preprocessing function
def preprocess_text(text):
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

# Read the dataset from the CSV file
df = pd.read_csv('dataset.csv')

# Apply preprocessing to the 'Post' column
df['Post'] = df['Post'].apply(preprocess_text)

# Text preprocessing and feature extraction using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Post'])

# Sentiment labels
y = df['Label Set']

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize multiple classifiers
classifiers = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': MultinomialNB(),
    'SVM': SVC()
}

# Store results of all algorithms
results = {}

# Train and evaluate each classifier
for name, clf in classifiers.items():
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_test)  
    accuracy = accuracy_score(y_test, y_pred)  
    results[name] = accuracy 
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")
    print(f"Classification Report for {name}:\n{classification_report(y_test, y_pred, zero_division=1)}\n")

# Find the best model
best_model_name = max(results, key=results.get)
best_model = classifiers[best_model_name]
print(f"Best Model: {best_model_name} with Accuracy: {results[best_model_name] * 100:.2f}%")

# Train the best model
best_model.fit(X_train, y_train)

# Home route to serve the HTML page
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction route
@app.post("/predict")
async def predict_sentiment(request: Request):
    form_data = await request.form()
    user_input = form_data.get("comment") 
    if not user_input:
        return JSONResponse(content={"error": "No input provided."}, status_code=400)
    
    processed_input = preprocess_text(user_input) 
    input_vector = vectorizer.transform([processed_input])  
    prediction = best_model.predict(input_vector)  
    sentiment = "Hate" if prediction[0] == 1 else "Not Hate"
    return JSONResponse(content={"predicted_sentiment": sentiment})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)