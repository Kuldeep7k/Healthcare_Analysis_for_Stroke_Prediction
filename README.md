# Stroke Patient Healthcare Analysis Using Machine Learning

## Overview

The primary goal of this project is to develop a machine learning model that classifies patient data to predict the likelihood of a stroke. By analyzing various health and demographic factors, the project aims to identify key predictors of stroke, providing healthcare professionals with valuable insights to support early intervention and preventive care. The analysis leverages data processing, visualization, and predictive modeling techniques to assess and predict stroke risk.

## Key Features
- Data cleaning and preprocessing using Python libraries like `NumPy` and `Pandas`.
- Data exploration and visualization with `Matplotlib` and `Seaborn`.
- Application of machine learning models such as `Logistic Regression` for stroke prediction.
- Evaluation of model performance using metrics such as accuracy, precision, recall, and F1 score.

## Technologies Used
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Project Structure

    stroke-patient-healthcare-analysis/
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
    cd stroke-patient-healthcare-analysis
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


## License
This project is licensed under the MIT License - see the LICENSE file for details.