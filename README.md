# Stroke Patient Healthcare Analysis Using Deep Learning

## Overview

Overview
The primary goal of this project was to develop a deep learning model to classify patient data and predict the likelihood of a stroke. By analyzing various health and demographic factors, the project aimed to identify key predictors of stroke, providing healthcare professionals with valuable insights to support early intervention and preventive care. The analysis leveraged data processing, visualization, and predictive modeling techniques to assess and predict stroke risk.

## Key Features
- Data cleaning and preprocessing using Python libraries like `NumPy` and `Pandas`.
- Data exploration and visualization with `Matplotlib` and `Seaborn`.
- Application of machine learning models such as Linear Regression, Lasso Regression, Ridge Regression, and Logistic Regression for stroke prediction.
- Evaluation of model performance using metrics such as accuracy, precision, recall, F1 score, and RMSE.

## Technologies Used
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Project Structure

    stroke_patient_healthcare_analysis/
    ├── stroke_data.csv              # Raw dataset used for analysis
    ├── stroke_analysis.ipynb        # Jupyter notebook for analysis, data processing, and model building
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

## Evaluation Metrics
The model's performance is evaluated using the following metrics:

 - Accuracy: The overall accuracy of the model in predicting correct stroke occurrences.
 - Precision: The proportion of true positive predictions out of all positive predictions made by the model.
 - Recall: The proportion of actual stroke cases correctly identified by the model.
 - F1 Score: The harmonic mean of precision and recall, providing a balanced evaluation.
 - RMSE (Root Mean Squared Error): A measure used to evaluate regression models by calculating the square root of the average squared differences between predicted and actual values.

# Conclusion
After evaluating multiple machine learning models, we found that Logistic Regression performed best in predicting stroke occurrences with high accuracy. However, during further evaluation, it was observed that the model was not correctly predicting stroke cases but instead falsely predicting non-stroke cases. This issue arose because the model was highly biased towards the majority class (non-stroke cases).

The primary goal of this project was to develop a deep learning model to classify patient data and predict the likelihood of a stroke. However, due to the small size of the dataset, we opted to build a machine learning model instead of a deep learning model. Given the limited dataset, a deep learning model might not have outperformed the machine learning approach, and the results would likely have been similar.

**Future improvements** may involve addressing this bias, such as using techniques like resampling, adjusting class weights, or exploring more advanced models.


## License
This project is licensed under the MIT License - [click here](https://github.com/Kuldeep7k/Healthcare_Analysis_for_Stroke_Prediction/blob/main/LICENSE)  see the LICENSE file for details.