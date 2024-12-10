import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np


class LogisticRegression:
    # Initialize learning rate (lr) and number of iterations (n_iters)
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr   # Learning rate (α)
        self.n_iters = n_iters  # Number of iterations for gradient descent
        self.weights = None  # Initialize weights (slopes)
        self.bias = None  # Initialize bias (intercept)

    # Sigmoid function to map predictions to probabilities between 0 and 1
    def sigmoid(self, z):
        # Applying the sigmoid function: y_pred = 1 / (1 + exp(-z))
        return 1 / (1 + np.exp(-z))

    # Function to train the logistic regression model using X_train and y_train
    def train(self, X, y): 
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)  # Initialize weights to zero
        self.bias = 0  # Initialize bias to zero

        for _ in range(self.n_iters):
            
            # Calculate the linear combination of inputs and weights: z = X * weights + bias
            model = np.dot(X, self.weights) + self.bias
            
            # Transforming the linear output into probabilities using the sigmoid function
            y_pred = self.sigmoid(model)

            # Gradient calculation for the weights (dw): 
            # dw = (1 / num_samples) * X.T * (y_pred - y)
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y)) 
            
            # Gradient calculation for the bias (db):
            # db = (1 / num_samples) * sum(y_pred - y)
            db = (1 / num_samples) * np.sum(y_pred - y) 

            # Update the weights and bias using the gradients and learning rate:
            # weights := weights - learning_rate * dw, bias := bias - learning_rate * db
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    # Function to predict labels based on input test data (X_test)
    def predict(self, X):
        # Calculate the linear output from the input data and learned weights: z = X * weights + bias
        linear_model = np.dot(X, self.weights) + self.bias 
        
        # Transform the linear output into predicted probabilities using the sigmoid function
        y_pred = self.sigmoid(linear_model)  # Applying sigmoid to get probabilities: y_pred = 1 / (1 + exp(-z))

        # Classify based on threshold 0.5: if probability >= 0.5, classify as 1, else 0
        predictions = []
        for p in y_pred:
            if p >= 0.5:
                predictions.append(1)  # Classify as 1 if probability is >= 0.5
            else:
                predictions.append(0)  # Classify as 0 if probability is < 0.5

        return np.array(predictions)  # Return the final binary predictions as an array
    
    # Function to calculate accuracy as the proportion of correct predictions using X_test and y_test
    def score(self, X, y):
        # Predict labels for the input data
        y_pred = self.predict(X) 
        
        # Calculate accuracy as the percentage of correct predictions
        accuracy = np.mean(y_pred == y)  # Accuracy = (correct predictions) / (total predictions)
        
        return accuracy 

    # Function to return predicted probabilities for the positive class (class 1) using X_test
    def predict_proba(self, X):
        # Compute the linear combination of inputs and weights
        linear_model = np.dot(X, self.weights) + self.bias 

        # Get probability of positive class (class 1)
        probabilities = self.sigmoid(linear_model) 

        # Return both probabilities (negative class, positive class) 
        return [(1 - p, p) for p in probabilities]  


# Set page configuration
st.set_page_config(page_title="Stroke Prediction Web App")

# Title of the Web Application
st.title("Stroke Prediction Web App")

st.markdown(
    """
    This application uses machine learning to predict the likelihood of a person having a stroke based on various health and lifestyle features.

    **How to use this app:**
    - Enter the relevant information in the input fields, such as age, BMI, gender, hypertension status, and more.
    - The app will predict the likelihood of stroke risk based on the provided data.
    - Once the prediction is made, the app will display the stroke risk percentage and its associated risk level.

    **Stroke Risk Levels:**
    - **High risk**: A stroke probability of 80% or more.
    - **Moderate risk**: A stroke probability between 50% and 80%.
    - **Low risk**: A stroke probability below 50%.

    **Important:** If the app shows a high risk of stroke, please seek immediate medical consultation.
    """
)


# Function to load the model
def load_model(model_filename='logistic_reg_model.pkl'):
    # Check if the model file exists
    if not os.path.exists(model_filename):
        st.error(f"Error: Model file '{model_filename}' not found!")
        return None, False

    # Load the model using joblib
    log_reg_model = joblib.load(model_filename)

    # Check if the model is loaded properly
    if log_reg_model is not None:
        st.success("Model loaded successfully!")
        return log_reg_model, True
    else:
        st.error("Error: Failed to load the model.")
        return None, False

# Attempt to load the model
log_reg_model, model_trained = load_model()

# Display whether the model was loaded successfully
if model_trained:
    st.write("Model is ready to make predictions!")
else:
    st.warning("Model not loaded. Please ensure the model file is accessible and try again.")

# User input fields on the main page
st.header("Enter the Input Features")

# Create two columns for input fields
col1, col2 = st.columns(2)

# First column for inputs
with col1:
    gender = st.radio("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, step=0.1)
    bmi = st.number_input("BMI", min_value=0.0, step=0.1)
    hypertension = st.radio("Hypertension (0: No, 1: Yes)", [0, 1])
    heart_disease = st.radio("Heart Disease (0: No, 1: Yes)", [0, 1])
    
    

# Second column for inputs
with col2:
    residence_type = st.radio("Residence Type", ["Urban", "Rural"])
    work_type = st.radio("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    smoking_status = st.radio("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])
    ever_married = st.radio("Ever Married (0: No, 1: Yes)", [0, 1])

# Validation to ensure all fields are filled
if st.button("Predict Stroke Risk"):
    if not all([age, avg_glucose_level, bmi]):
        st.warning("Please fill in all the input fields!")
    else:
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
        input_df = pd.DataFrame([input_data])

        # Get prediction probability for stroke (class 1)
        prediction_proba = log_reg_model.predict_proba(input_df)
        stroke_probability = prediction_proba[0][1] * 100  # Extract the probability for the positive class
        st.write(f"### Stroke Probability: {stroke_probability:.2f}%")

        if stroke_probability >= 80:
            st.error(f"High risk of stroke detected! Recommend medical consultation.")
        elif 50 <= stroke_probability < 80:
            st.warning(f"Moderate risk of stroke.")
        else:
            st.success(f"Low risk of stroke detected.")
