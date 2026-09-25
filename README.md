# Voting Ensemble Learning

A Machine Learning project that uses a **Voting Ensemble Classifier** to predict the species of an Iris flower based on its sepal and petal measurements.

The project combines multiple machine learning models and uses **Pipeline** to handle preprocessing and model training in a clean and reproducible way.

## 📌 Project Overview

The model predicts the Iris flower species using four input features:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

### Target

The target variable is:

* Setosa
* Versicolor
* Virginica

## 🤖 Models Used

The Voting Ensemble combines the following models:

1. **Logistic Regression**
2. **Support Vector Machine (SVM)**
3. **Decision Tree Classifier**

The final prediction is generated using **Soft Voting**, where the class probabilities from the individual models are combined.

```text
                 Iris Dataset
                      ↓
              Train / Test Split
                      ↓
       ┌──────────────┼──────────────┐
       ↓              ↓              ↓
 Logistic Regression  SVM      Decision Tree
       ↓              ↓              ↓
    Pipeline         Pipeline       Pipeline
       └──────────────┼──────────────┘
                      ↓
               Voting Classifier
                      ↓
                Final Prediction
                      ↓
                Iris Species
```

## 🔧 Technologies Used

* Python
* Pandas
* Scikit-learn
* Jupyter Notebook / Google Colab
* Git & GitHub

## 📂 Dataset

The project uses the **Iris dataset**.

The dataset contains:

* **150 samples**
* **4 numerical features**
* **3 target classes**

### Features

| Feature        | Description         |
| -------------- | ------------------- |
| `sepal_length` | Length of the sepal |
| `sepal_width`  | Width of the sepal  |
| `petal_length` | Length of the petal |
| `petal_width`  | Width of the petal  |

### Target

```text
species
```

## ⚙️ Preprocessing

`LabelEncoder` is used to convert the categorical target labels into numerical values.

```python
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
```

For models that benefit from feature scaling, `StandardScaler` is included inside the Pipeline.

## 🔗 Pipeline

Each model can have its own preprocessing pipeline.

Example:

```python
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])
```

Using Pipeline helps ensure that preprocessing and model training are performed consistently.

## 🗳️ Voting Ensemble

The project uses `VotingClassifier` from Scikit-learn.

```python
voting_model = VotingClassifier(
    estimators=[
        ("lr", lr_pipeline),
        ("svm", svm_pipeline),
        ("ds", ds_pipeline)
    ],
    voting="soft"
)
```

### Soft Voting

Soft voting combines the predicted probabilities of all models and selects the class with the highest combined probability.

## 📊 Evaluation Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score

Example:

```python
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
```

## 📈 Visualization

The project also includes visualizations such as:

* Scatter plots
* Feature relationships
* Confusion matrix
* Model accuracy comparison

Example scatter plot:

```python
plt.scatter(
    df["sepal_length"],
    df["sepal_width"],
    c=y_encoded
)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris Dataset")
plt.show()
```

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/mi-ganesh/Voter_Ensemble_Learning.git
```

### 2. Open the project

```bash
cd Voter_Ensemble_Learning
```

### 3. Install required libraries

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

### 4. Run the notebook

Open:

```text
Voter_ensembel.ipynb
```

You can run it using Jupyter Notebook or upload it to Google Colab.

## 📁 Project Structure

```text
Voter_Ensemble_Learning/
│
├── Voter_ensembel.ipynb
├── iris.csv
├── .gitignore
└── README.md
```

## 🎯 Learning Objectives

This project demonstrates:

* Classification using Machine Learning
* Label Encoding
* Train-Test Split
* Feature Scaling
* Machine Learning Pipelines
* Logistic Regression
* Support Vector Machine
* Decision Tree
* Ensemble Learning
* Voting Classifier
* Soft Voting
* Model Evaluation
* Data Visualization

## 👨‍💻 Author

**Ganesh Shidwadkar**

GitHub: [mi-ganesh](https://github.com/mi-ganesh)

## ⭐ Future Improvements

* Compare Hard Voting and Soft Voting
* Tune individual model hyperparameters
* Compare Voting Ensemble with Random Forest
* Add cross-validation
* Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV
* Build a simple web interface for predictions
