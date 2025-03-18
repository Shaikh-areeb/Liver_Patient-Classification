# Indian Liver Patient Classification

## Project Overview
This project focuses on classifying Indian liver patients based on medical attributes. The dataset contains records of patients diagnosed with liver disease and those who are not. The goal is to develop a machine learning model to predict whether a patient has liver disease.

## Dataset
- **Source:** UCI Machine Learning Repository (Indian Liver Patient Dataset)
- **Features:**
  - Age
  - Gender
  - Total Bilirubin
  - Direct Bilirubin
  - Alkaline Phosphotase
  - Alamine Aminotransferase (ALT)
  - Aspartate Aminotransferase (AST)
  - Total Proteins
  - Albumin
  - Albumin and Globulin Ratio
  - **Label:** (1 = Liver Disease, 0 = No Disease)

## Objective
- Perform **Exploratory Data Analysis (EDA)** to identify patterns in the dataset.
- Handle **missing values and outliers** if present.
- Train **classification models** to predict liver disease status.
- Evaluate models using accuracy, precision, recall, and F1-score.

## Technologies Used
- **Python**
- **Pandas, NumPy** (Data Preprocessing)
- **Matplotlib, Seaborn** (Data Visualization)
- **Scikit-learn** (Machine Learning)

## Model Training
The dataset is split into **training and testing sets**, and various classification models are trained, including:
- Decision Tree
- Random Forest

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Results
The best-performing model is selected based on its **classification accuracy and other evaluation metrics**.

## How to Run
1. Install the required libraries using:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
   ```
2. Run the preprocessing script to clean and prepare the data.
3. Train the model using the Jupyter Notebook or Python script provided.
4. Evaluate the model and generate predictions.

## Conclusion
This project provides insights into liver disease classification and highlights the effectiveness of different machine learning models in medical diagnosis.



