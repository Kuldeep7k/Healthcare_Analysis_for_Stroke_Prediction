# Stroke Patient Healthcare Analysis Using Deep Learning

## Overview

The primary goal of this project is to develop a deep learning model to classify patient data and predict the likelihood of a stroke. By analyzing various health and demographic factors, the project aims to identify key predictors of stroke, providing healthcare professionals with valuable insights to support early intervention and preventive care. The analysis leverages data processing, visualization, predictive modeling, and a user-friendly web application built with Streamlit for interactive analysis and predictions.

## Key Features
- Data cleaning and preprocessing using Python libraries like `NumPy` and `Pandas`.
- Data exploration and visualization with `Matplotlib` and `Seaborn`.
- Application of machine learning models such as Linear Regression, Lasso Regression, Ridge Regression, and Logistic Regression for stroke prediction.
- Evaluation of model performance using metrics such as accuracy, precision, recall, F1 score, and RMSE.
- Interactive web-based interface using Streamlit for easy data visualization, exploration, and model predictions.

## Technologies Used
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- joblib

## Project Structure

    stroke_patient_healthcare_analysis/
    ├── .streamlit/                  # Streamlit configuration directory
        ├── config.toml              # Streamlit configuration file
    ├── stroke_analysis.ipynb        # Jupyter notebook for analysis, data processing, and model building
    ├── streamlit_app.py             # Streamlit app for interactive predictions
    ├── logistic_reg_model.pkl       # Serialized logistic regression model
    ├── requirements.txt             # Python dependencies
    └── README.md                    # This file


## Installation

### To run this project locally, follow the steps below:

1.  Clone the repository:

    ```git
    git clone <repository-url>
    ```
    
2.  Navigate into the project directory:
    ```cmd
    cd stroke_patient_healthcare_analysis
    ```

3.  Install required packages:
    ```python
    pip install -r requirements.txt
    ```


## Usage

### Running the Jupyter Notebook

Load the dataset and perform initial data exploration and cleaning by running the Jupyter notebook stroke_analysis.ipynb.

To open the notebook, you can run:

```bash
jupyter notebook stroke_analysis.ipynb
```

The notebook will guide you through the process of:

 - Loading the dataset (stroke_data.csv)
 - Data cleaning and preprocessing
 - Data exploration and visualization
 - Building and evaluating a machine learning model (Logistic Regression)
 - You can also train and evaluate the model directly within the notebook, using various evaluation metrics like accuracy, precision, recall, and F1 score.

### Running the Streamlit App

To interactively make predictions, you can run the Streamlit app:

Start the app:

```cmd
streamlit run streamlit_app.py
```

Open the link provided in the terminal to access the web application.

The app includes:
    
- A prediction interface where users can input patient details and receive a stroke risk prediction.

## Evaluation Metrics
The model's performance is evaluated using the following metrics:

 - Accuracy: The overall accuracy of the model in predicting correct stroke occurrences.
 - Precision: The proportion of true positive predictions out of all positive predictions made by the model.
 - Recall: The proportion of actual stroke cases correctly identified by the model.
 - F1 Score: The harmonic mean of precision and recall, providing a balanced evaluation.
 - RMSE (Root Mean Squared Error): A measure used to evaluate regression models by calculating the square root of the average squared differences between predicted and actual values.

# Conclusion
After evaluating multiple machine learning models, we found that Logistic Regression performed best in predicting stroke occurrences with high accuracy. However, during further evaluation, it was observed that the model was not correctly predicting stroke cases but instead falsely predicting non-stroke cases. This issue arose because the model was highly biased towards the majority class (non-stroke cases).

## Future Improvements:

- Addressing the class imbalance using techniques like resampling, adjusting class weights, or exploring advanced models.
- Incorporating additional patient features to enhance model accuracy.
- Enhancing the Streamlit app to include interactive feature importance visualizations and real-time data analysis.


## License
This project is licensed under the MIT License - [click here](https://github.com/Kuldeep7k/Healthcare_Analysis_for_Stroke_Prediction/blob/main/LICENSE)  see the LICENSE file for details.