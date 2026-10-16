# 🏥 Medical Insurance Cost Prediction Using Machine Learning

## 📌 Project Overview

This project focuses on **Exploratory Data Analysis (EDA)** and Machine Learning to predict individual medical insurance charges based on personal and lifestyle-related features such as age, BMI, smoking status, number of children, sex, and residential region.

The goal is to explore the dataset, identify important patterns, train regression models, and compare their performance to select a suitable model for predicting insurance costs.

## 🎯 Objectives

- Perform exploratory data analysis on medical insurance data.
- Identify missing values, duplicate records, distributions, and outliers.
- Visualize relationships between insurance charges and other features.
- Encode categorical variables for machine learning.
- Train and compare multiple regression models.
- Tune Random Forest hyperparameters.
- Evaluate models using MAE, RMSE, and R².
- Save the trained model for future predictions.

## 📂 Dataset Information

The project uses the Medical Cost Personal Dataset, commonly available on Kaggle.

**Dataset:** `insurance.csv`

| Feature | Description |
|---|---|
| `age` | Age of the individual |
| `sex` | Sex of the individual |
| `bmi` | Body Mass Index |
| `children` | Number of dependent children |
| `smoker` | Smoking status |
| `region` | Residential region |
| `charges` | Medical insurance charges (target variable) |

Dataset source: [Medical Cost Personal Dataset on Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance)

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

## 🔍 Exploratory Data Analysis (EDA)

The following analysis was performed:

1. Dataset inspection using `head()`, `shape`, `info()`, and `describe()`.
2. Missing-value and duplicate checks.
3. Statistical summaries of numerical and categorical features.
4. Histograms and boxplots to examine distributions and potential outliers.
5. Count plots for categorical variables.
6. Analysis of insurance charges by smoking status.
7. Relationship analysis between age, BMI, and insurance charges.
8. Correlation heatmap for numerical features.

## ⚙️ Data Preprocessing

- Checked missing values and duplicate records.
- Converted categorical variables into numerical features using one-hot encoding.
- Separated the input features (`X`) and target variable (`y`).
- Split the dataset into training and testing sets using an 80:20 ratio.

## 🤖 Machine Learning Models

The following regression models were trained and evaluated:

1. Linear Regression
2. Random Forest Regressor
3. Tuned Random Forest Regressor using `RandomizedSearchCV`

## 📊 Model Performance

The following results were obtained on the test dataset:

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 4,181.194 | 5,796.285 | 0.784 |
| Random Forest | 2,489.713 | 4,451.563 | 0.872 |
| Tuned Random Forest | 2,506.871 | 4,359.191 | 0.878 |

### 📈 Evaluation Metrics

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted insurance charges. Lower is better.
- **RMSE (Root Mean Squared Error):** Penalizes larger prediction errors more strongly. Lower is better.
- **R² Score:** Measures how much variation in the target is explained by the model on the evaluation dataset. Higher is generally better.

### 🏆 Best Model

The **Tuned Random Forest Regressor** achieved the highest R² score (0.878) and the lowest RMSE (4,359.191) among the evaluated models.

The original Random Forest achieved a slightly lower MAE (2,489.713) than the tuned model (2,506.871). Therefore, the preferred model depends on which evaluation metric matters most for the intended application.

## 💾 Saving the Model

The trained model was saved using Joblib:

```python
import joblib

joblib.dump(tuned_model, "insurance_model.pkl")
```

Load the saved model:

```python
loaded_model = joblib.load("insurance_model.pkl")
```

**Note:** Predictions on new data require the same feature preprocessing and column order used during training.

## 🚀 Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```

### 2. Navigate to the project directory

```bash
cd YOUR-REPOSITORY
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

Alternatively, install dependencies from a `requirements.txt` file if one is included:

```bash
pip install -r requirements.txt
```

### 4. Run the notebook

```bash
jupyter notebook
```

Open the project notebook and run the cells in order.

## 📁 Suggested Project Structure

```text
insurance-cost-prediction/
│
├── data/
│   └── insurance.csv
│
├── notebooks/
│   └── insurance_eda_ml.ipynb
│
├── insurance_model.pkl
├── requirements.txt
└── README.md
```

Adjust this structure to match the actual files in your repository.

## 🔮 Future Improvements

- Build an interactive prediction interface using Streamlit.
- Apply permutation importance to investigate feature importance.
- Use cross-validation for more robust model comparison.
- Improve prediction accuracy through additional feature engineering and hyperparameter optimization.
- Deploy the prediction application online.

## ⚠️ Disclaimer

This project is intended for educational and demonstration purposes only. Predictions are estimates
