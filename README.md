# 🏦 Loan Approval Predictor

This project builds a machine learning pipeline to predict whether a loan will be approved based on applicant details. We use both **Logistic Regression** and a **tuned Decision Tree Classifier** to evaluate and compare performance. The project includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, and prediction for new user input.

---

## 📁 Dataset

The project uses the **Loan Prediction Dataset** from Kaggle which contains the following columns:

- **Loan_ID**: Unique Loan ID
- **Gender**: Male/Female
- **Married**: Applicant’s marital status
- **Dependents**: Number of dependents
- **Education**: Graduate/Not Graduate
- **Self_Employed**: Self employed or not
- **ApplicantIncome**: Income of the applicant
- **CoapplicantIncome**: Income of the coapplicant
- **LoanAmount**: Loan amount in thousands
- **Loan_Amount_Term**: Term of the loan in months
- **Credit_History**: Credit history meets guidelines
- **Property_Area**: Urban/Semi-Urban/Rural
- **Loan_Status** (only in `train.csv`): Target variable (Y/N)

---

## 🔧 Project Structure

```
Loan Approval Predictor.ipynb
train.csv
test.csv
README.md
```

---

## 🧹 Step 1: Data Preprocessing

- Loaded and inspected the dataset
- Handled missing values
- Applied label encoding and one-hot encoding where necessary
- Standardized numerical features using `StandardScaler`
- Split the data into training and test sets

---

## 📊 Step 2: Model Building - Logistic Regression

- Fitted logistic regression on the training set
- Performed 5-fold cross-validation
- Evaluated using accuracy, confusion matrix, and classification report
- ROC-AUC curve plotted

**Final Accuracy:** ~86.18%

---

## 🌲 Step 3: Model Building - Decision Tree (Tuned)

- Used `GridSearchCV` for hyperparameter tuning (`max_depth`, `min_samples_split`, etc.)
- Trained the best estimator on the training set
- Evaluated performance on the test set

**Best Parameters:**  
```python
{'class_weight': None, 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2}
```

**Final Accuracy:** ~84.55%

---

## 📈 Step 4: Evaluation & Comparison

- Compared Logistic Regression and Decision Tree models using:
  - Accuracy
  - Confusion Matrix
  - ROC-AUC Curve
  - Classification Report

---

## 🧪 Step 5: Prediction on Test Set

- Loaded and preprocessed `test.csv` with the same pipeline
- Predicted loan approval status using the tuned Decision Tree model
- Saved predictions in CSV format

---

## 👤 Step 6: Guided User Input for Real-time Prediction

- Allowed user to enter values manually via prompts
- Converted input into a format compatible with model expectations
- Performed real-time prediction using both models

Example:
```
Enter Gender (Male/Female): Male
Enter Married status (Yes/No): No
...
Prediction: Loan will be Approved ✅
```

---

## 📌 Key Libraries Used

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`: `LogisticRegression`, `DecisionTreeClassifier`, `GridSearchCV`, `StandardScaler`, `train_test_split`

---

## 📊 Model Performance Snapshot

| Metric             | Logistic Regression | Decision Tree (Tuned) |
|--------------------|---------------------|------------------------|
| Accuracy           | 86.18%              | 84.55%                 |
| Precision (Class 1)| High                | Very High              |
| ROC-AUC            | High                | High                   |

---

## 🚀 Future Improvements

- Integrate Flask/FastAPI for web interface
- Add XGBoost and RandomForest models
- Address class imbalance more deeply
- Enable bulk user input (CSV upload)

---

## 🧠 Learning Outcomes

- End-to-end supervised ML workflow
- Model tuning with cross-validation
- Practical insights into decision tree vs. logistic regression
- Real-world user input handling

---

## 🙌 Acknowledgements

- Dataset: [Kaggle - Loan Prediction Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- Tools: Python, Jupyter Notebook, Scikit-learn

---

## 📝 Author

*Developed by a beginner in machine learning as part of a structured learning journey.*