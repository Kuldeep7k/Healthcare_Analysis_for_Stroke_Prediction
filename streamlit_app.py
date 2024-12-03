import streamlit as st
import pandas as pd
import joblib
import warnings
import os
import numpy as np

# Suppress warnings to keep the output clean
warnings.filterwarnings('ignore')

# Custom Logistic Regression Class
class LogisticRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr  # Learning rate
        self.n_iters = n_iters  # Number of iterations
        self.weights = None  # Weights
        self.bias = None  # Bias

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def train(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.n_iters):
            model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(model)

            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        return (y_pred >= 0.5).astype(int)

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        probabilities = self.sigmoid(linear_model)
        return [(1 - p, p) for p in probabilities]

# Set up Streamlit app
st.set_page_config(page_title="Stroke Prediction Web App")

# Title of the Web Application
st.title("Stroke Prediction Web App")

st.markdown(
    """
    This application uses a custom machine learning model to predict the likelihood of stroke.
    **Steps to use:**
    1. Enter the required input fields in the sidebar.
    2. Click 'Predict Stroke Risk' to view results.
    """
)

# Function to load the trained model
def load_model(model_filename='logistic_reg_model.pkl'):
    if not os.path.exists(model_filename):
        st.error(f"Error: Model file '{model_filename}' not found!")
        return None, False

    try:
        # Load the model using joblib
        log_reg_model = joblib.load(model_filename)
        if isinstance(log_reg_model, LogisticRegression):
            st.success("Model loaded successfully!")
            return log_reg_model, True
        else:
            st.error("Error: Loaded object is not a LogisticRegression instance.")
            return None, False
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, False

# Attempt to load the model
log_reg_model, model_trained = load_model()

# Load the dataset and preprocess it
@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?export=download&id=1XyhVIZaKYZczlM2alun_fofilqTBq_9c"
    df = pd.read_csv(url)

    # Data Cleaning and Encoding
    df['bmi'].fillna(df['bmi'].median(), inplace=True)
    df['gender'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)
    df['ever_married'] = df['ever_married'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['Residence_type'] = df['Residence_type'].apply(lambda x: 1 if x == 'Urban' else 0)
    df = pd.get_dummies(df, columns=['work_type', 'smoking_status'], drop_first=False)

    # Transform specified columns to integer type
    columns_to_transform = [
        'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private',
        'work_type_Self-employed', 'work_type_children', 'smoking_status_Unknown',
        'smoking_status_formerly smoked', 'smoking_status_never smoked',
        'smoking_status_smokes'
    ]
    df[columns_to_transform] = df[columns_to_transform].astype(int)

    # Split data into features and target
    X = df.drop(columns=['id', 'stroke'])
    y = df['stroke']
    return df, X, y

# Load the data
df, X, y = load_data()

# Display dataset preview
st.write("### Dataset Preview")
st.dataframe(df.head())

# Sidebar inputs
st.sidebar.header("Input Features")

# Sidebar columns for inputs
col1, col2 = st.sidebar.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    hypertension = st.selectbox("Hypertension (0: No, 1: Yes)", [0, 1])
    heart_disease = st.selectbox("Heart Disease (0: No, 1: Yes)", [0, 1])
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, step=0.1)
    bmi = st.number_input("BMI", min_value=0.0, step=0.1)
    gender = st.selectbox("Gender", ["Male", "Female"])

with col2:
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])
    ever_married = st.selectbox("Ever Married (0: No, 1: Yes)", [0, 1])

# Prepare input data for prediction
input_data = {
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,
    "gender": 1 if gender == "Male" else 0,
    "ever_married": ever_married,
    "Residence_type": 1 if residence_type == "Urban" else 0,
    "work_type_Private": 1 if work_type == "Private" else 0,
    "work_type_Self-employed": 1 if work_type == "Self-employed" else 0,
    "work_type_Govt_job": 1 if work_type == "Govt_job" else 0,
    "work_type_children": 1 if work_type == "children" else 0,
    "work_type_Never_worked": 1 if work_type == "Never_worked" else 0,
    "smoking_status_formerly smoked": 1 if smoking_status == "formerly smoked" else 0,
    "smoking_status_never smoked": 1 if smoking_status == "never smoked" else 0,
    "smoking_status_smokes": 1 if smoking_status == "smokes" else 0,
    "smoking_status_Unknown": 1 if smoking_status == "Unknown" else 0,
}

# Convert the input data into a DataFrame
input_df = pd.DataFrame([input_data], columns=X.columns).fillna(0)

# Prediction button
if st.button("Predict Stroke Risk"):
    if not model_trained:
        st.warning("Model is not loaded. Please train or load the model.")
    else:
        # Predict probabilities
        prediction_proba = log_reg_model.predict_proba(input_df)
        stroke_probability = prediction_proba[0][1] * 100
        st.write(f"### Stroke Probability: {stroke_probability:.2f}%")

        # Show risk level
        if stroke_probability >= 50:
            st.error("High risk of stroke detected! Consult a healthcare professional.")
        else:
            st.success("Low risk of stroke detected.")
