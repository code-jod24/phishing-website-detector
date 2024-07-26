# Import necessary libraries
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# Load the dataset
data = pd.read_csv('phishing_website_dataset.csv')

# Convert column names to lowercase to handle potential case sensitivity issues
data.columns = data.columns.str.lower()

# Exploratory Data Analysis (EDA)
print(data.head())
print(data.info())
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Visualize the distribution of the target variable
fig = px.histogram(data, x='label', title='Distribution of Legitimate and Phishing URLs', labels={'label':'Label'})
fig.show()

# Visualize relationships between features and the target variable
fig = px.box(data, x='label', y='urllength', title='URL Length vs Label', labels={'label':'Label', 'urllength':'URL Length'})
fig.show()

# Feature selection and engineering
selected_features = ['urllength', 'domainlength', 'isdomainip', 'tldlegitimateprob', 'charcontinuationrate']
X = data[selected_features]
y = data['label']

# Handle categorical data if necessary
# Assume 'isdomainip' is categorical and rest are numerical

# Preprocessing
numerical_features = ['urllength', 'domainlength', 'tldlegitimateprob', 'charcontinuationrate']
categorical_features = ['isdomainip']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compute class weights to handle imbalance
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i : class_weights[i] for i in range(len(class_weights))}

# Model training with class weights
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weights))])

model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model
import joblib
joblib.dump(model, 'phishing_url_detector.pkl')

# Load the model (for demonstration)
# loaded_model = joblib.load('phishing_url_detector.pkl')
# loaded_model.predict(X_test)