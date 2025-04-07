import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import mlflow

# Set environment variable
os.environ['REQUESTS_CA_BUNDLE'] = r'C:\Users\LENOVO\anaconda3\Lib\site-packages\certifi\cacert.pem'
print(os.environ.get('REQUESTS_CA_BUNDLE'))

# Set MLflow tracking URI and experiment
mlflow.set_tracking_uri("https://mlflow.cs.ait.ac.th/")  # Replace with your tracking URI
mlflow.set_experiment("st125675-a3")  # Replace with your experiment name

# Load data
data = pd.read_csv("cars.csv")

# Preprocessing
data.dropna(subset=['selling_price'], inplace=True)
data['owner'].fillna(data['owner'].mode()[0], inplace=True)
data.drop(data[data['selling_price'] == 0].index, inplace=True)
data['seller_type'] = data['seller_type'].apply(lambda x: 1 if x == 'Individual' else 0)
data['transmission'] = data['transmission'].apply(lambda x: 1 if x == 'Manual' else 0)
data['mileage'] = data['mileage'].str.split(' ').str[0].astype(float)
data['engine'] = data['engine'].str.split(' ').str[0].astype(float)
data['max_power'] = data['max_power'].str.split(' ').str[0].astype(float)
data.drop(data[data['year'] > 2023].index, inplace=True)
data.drop(data[data['selling_price'] > 10000000].index, inplace=True)
data.drop(data[data['engine'] > 5000].index, inplace=True)
data.drop(data[data['mileage'] > 500].index, inplace=True)
data.drop(data[data['max_power'] > 400].index, inplace=True)
data.drop(data[data['torque'] == 'null'].index, inplace=True)
data['torque'] = data['torque'].str.split(' ').str[0]
data['torque'] = data['torque'].str.split('nm').str[0]
data['torque'] = data['torque'].str.split('kgm').str[0]
data['torque'] = data['torque'].astype(float)
data.drop('name', axis=1, inplace=True)

# Define price buckets
data['price_bucket'] = pd.qcut(data['selling_price'], q=4, labels=[0, 1, 2, 3])
X = data.drop(['selling_price', 'price_bucket'], axis=1)
y = data['price_bucket']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define categorical and numerical features
categorical_features = ['fuel', 'seller_type', 'transmission', 'owner']
numerical_features = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque']

# Create transformers
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model_name", "RidgeLogisticRegression")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)

    # Create and train Ridge Logistic Regression model
    model = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', LogisticRegression(solver='liblinear', multi_class='ovr', penalty='l2', C=0.1))])  # Add L2 regularization
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            mlflow.log_metric(f"precision_{label}", metrics['precision'])
            mlflow.log_metric(f"recall_{label}", metrics['recall'])
            mlflow.log_metric(f"f1-score_{label}", metrics['f1-score'])
            mlflow.log_metric(f"support_{label}", metrics['support'])

    # Log the trained model
    mlflow.sklearn.log_model(model, "ridge_logistic_regression_model")

    print("MLflow run completed!")
