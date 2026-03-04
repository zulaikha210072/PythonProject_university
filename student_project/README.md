Student Performance Prediction Using Linear Regression (From Scratch)
Project Overview

This project implements a complete end-to-end predictive modeling pipeline to estimate students’ mathematics scores using Linear Regression built entirely from scratch with NumPy.
The objective was not only to build a predictive model, but also to deeply understand the mathematical foundations of linear regression, gradient descent optimization, feature normalization, 
and model evaluation — without relying on high-level machine learning libraries such as scikit-learn.

The project emphasizes clarity, modularity, and mathematical correctness.

Objectives
- Build a regression model without using machine learning frameworks 
- Understand and implement gradient descent manually 
- Apply proper data preprocessing techniques 
- Evaluate model performance using standard regression metrics 
- Visualize training behavior and prediction performance

Technical Stack 
- Python 3.10
- NumPy (numerical computation)
- Pandas (data handling)
- Matplotlib (visualization)
- No machine learning libraries (e.g., sklearn) were used in model training.

Methodology
1. Data Preprocessing

Loaded dataset and handled missing values

Encoded categorical variables (e.g., gender, lunch type, test preparation)

Normalized numerical features to improve gradient descent stability

Split data into training and testing sets

2. Exploratory Data Analysis

Computed descriptive statistics

Generated correlation matrix

Visualized feature relationships using a heatmap

3. Model Implementation (From Scratch)

The model follows the standard linear regression formulation:

Hypothesis Function:
  h(x)=Xw+b 
  Where:
   X = feature matrix
   w = weight vector 
   b = bias term

 Used Cost Function (Mean Squared Error). 
 Gradient Descent Updates:
 The training loop iteratively updates parameters until convergence, while tracking cost reduction over epochs.

4. Model Evaluation

The trained model was evaluated using:

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

R² Score

Additionally:
Cost vs. Epoch curve was plotted to verify convergence
Predicted vs. Actual scatter plot was generated to visualize model performance

Project Structure
student_performance_project/
├── data/
│   └── StudentsPerformance.csv
├── src/
│   ├── preprocessing.py
│   ├── statistics.py
│   ├── regression.py
│   ├── evaluation.py
│   └── visualization.py
├── main.py
├── requirements.txt
└── README.md

The modular structure improves readability, maintainability, and debugging efficiency.
Key Learning Outcomes
- Strengthened understanding of linear algebra in machine learning 
- Implemented gradient descent manually 
- Practiced feature scaling and its impact on convergence 
- Improved debugging skills by stepping through iterative optimization 
- Reinforced knowledge of regression evaluation metrics

Conclusion :
This project demonstrates both theoretical understanding and practical implementation of
linear regression without reliance on automated machine learning libraries. It reflects a 
strong foundation in core machine learning principles and numerical computation.