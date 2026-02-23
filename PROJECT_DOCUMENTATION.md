# **Prosperity Prognosticator — Startup Success Prediction**

# **Project Description:**

The startup world is unpredictable. Investors pour millions into companies that never make it, while promising ideas die quietly for lack of funding. One of the biggest challenges in the startup ecosystem is figuring out — before it's too late — whether a venture has what it takes to survive and get acquired, or if it is heading toward a shutdown.

This project addresses that exact problem. By studying real-world startup data sourced from Crunchbase, we built a machine learning model that looks at patterns in funding history, investor relationships, milestones, and geography to predict whether a startup will ultimately succeed or fail. The idea is to give entrepreneurs and investors a data-driven signal — not a crystal ball, but a reasonable, evidence-backed assessment.

We trained and compared four classification algorithms — Logistic Regression, Support Vector Machine, Decision Tree, and Random Forest — and selected the best-performing one after careful hyperparameter tuning. The final model is saved and served through a Flask web application with a clean, user-friendly interface that anyone (not just a data scientist) can actually use.

# **Technical Architecture:**

```
[ User fills the web form ]
         |
         v
[ Flask Backend — app.py ]
         |
         v
[ StandardScaler (pre-processing) ]
         |
         v
[ Random Forest Classifier (.pkl model) ]
         |
         v
[ Prediction: SUCCESS / FAILURE + Confidence % ]
         |
         v
[ Result displayed on the UI ]
```

The dataset flows from raw CSV → EDA notebook → pre-processing → model training → saved pickle file → Flask API → HTML frontend.

# **Pre-requisites:**

**To complete this project, you must have the following software, concepts, and packages installed.**

* **Python Environment (VS Code or Jupyter Notebook):**

  * Install Python 3.x from [https://www.python.org/downloads/](https://www.python.org/downloads/)

  * Or use Anaconda: [https://www.anaconda.com/download](https://www.anaconda.com/download)

* **Python Packages:**

  * Open your terminal or Anaconda prompt.

  * Type `pip install numpy` and press Enter.

  * Type `pip install pandas` and press Enter.

  * Type `pip install scikit-learn` and press Enter.

  * Type `pip install matplotlib` and press Enter.

  * Type `pip install seaborn` and press Enter.

  * Type `pip install Flask` and press Enter.

  * Or simply install everything at once using:

    ```
    pip install -r requirements.txt
    ```

# **Prior Knowledge:**

You should have some familiarity with the following topics before going through this project.

* **ML Concepts**

  * Supervised learning: [https://www.javatpoint.com/supervised-machine-learning](https://www.javatpoint.com/supervised-machine-learning)

  * Classification vs Regression: [https://www.javatpoint.com/regression-vs-classification-in-machine-learning](https://www.javatpoint.com/regression-vs-classification-in-machine-learning)

  * Logistic Regression: [https://www.javatpoint.com/logistic-regression-in-machine-learning](https://www.javatpoint.com/logistic-regression-in-machine-learning)

  * Decision Tree: [https://www.javatpoint.com/machine-learning-decision-tree-classification-algorithm](https://www.javatpoint.com/machine-learning-decision-tree-classification-algorithm)

  * Random Forest: [https://www.javatpoint.com/machine-learning-random-forest-algorithm](https://www.javatpoint.com/machine-learning-random-forest-algorithm)

  * Support Vector Machine: [https://www.javatpoint.com/machine-learning-support-vector-machine-algorithm](https://www.javatpoint.com/machine-learning-support-vector-machine-algorithm)

  * GridSearchCV and Hyperparameter Tuning: [https://scikit-learn.org/stable/modules/grid_search.html](https://scikit-learn.org/stable/modules/grid_search.html)

  * Evaluation metrics: [https://www.analyticsvidhya.com/blog/2019/08/11-important-model-evaluation-error-metrics/](https://www.analyticsvidhya.com/blog/2019/08/11-important-model-evaluation-error-metrics/)

* **Flask Basics**: [https://www.youtube.com/watch?v=lj4I\_CvBnt0](https://www.youtube.com/watch?v=lj4I_CvBnt0)

# **Project Objectives:**

By the end of this project you will:

* Understand how real-world startup data is structured and what features drive success or failure.

* Be able to perform end-to-end exploratory data analysis — from loading the dataset all the way to drawing meaningful business insights through charts.

* Know how to clean and pre-process messy, real-world data — handling missing values, encoding targets, dropping irrelevant columns, and scaling features.

* Have hands-on experience training, comparing, and tuning multiple classification models.

* Know how to evaluate models properly using accuracy, precision, recall, F1-score, and confusion matrices.

* Be able to identify the most important features a model relies on and use them to build a leaner, faster final model.

* Know how to save a trained model using pickle and build a working Flask web API around it.

# **Project Flow:**

* User opens the web application and fills in details about their startup.

* The entered values are sent to the Flask backend via a POST request.

* The backend loads the saved Random Forest model and pre-processes the inputs using the same StandardScaler used during training.

* The model returns a prediction — **SUCCESS** or **FAILURE** — along with a confidence percentage.

* The result is displayed back on the UI in a clear, readable format.

To accomplish this, we completed all the following activities:

* Data collection

  * Downloaded the Crunchbase startup dataset from Kaggle

* Visualizing and analyzing data

  * Univariate analysis

  * Bivariate analysis

  * Multivariate analysis

  * Statistical / Descriptive analysis

* Data pre-processing

  * Checking for null values and handling missing data

  * Reducing category cardinality

  * Dropping irrelevant columns

  * Encoding the target variable

  * Scaling features with StandardScaler

  * Splitting data into train and test sets

* Model building

  * Training four classification algorithms

  * Evaluating and comparing model performance

  * Hyperparameter tuning with GridSearchCV

  * Feature importance analysis and optimal feature selection

  * Saving the final model as a .pkl file

* Application Building

  * Creating the HTML frontend

  * Writing the Flask backend (app.py)

  * Running and testing the full application

# **Project Structure:**

```
startup_pred/
├── templates/
│   └── index.html              ← Web UI (form, tooltips, INR converter, city quick-fill)
├── app.py                      ← Flask backend + /predict REST API
├── random_forest_model.pkl     ← Trained model (scaler + selected features bundled)
├── startup data.csv            ← Raw Crunchbase dataset from Kaggle
├── startup-prediction-eda-model.ipynb  ← Full EDA + model training notebook
├── requirements.txt            ← All Python dependencies
└── README.md                   ← Project overview and setup guide
```

* We are building a Flask application that serves an HTML page stored in the `templates/` folder. All prediction logic lives in `app.py`.

* `random_forest_model.pkl` is our saved model. It stores not just the model, but also the scaler and the list of features needed — so the Flask app can load everything from one file.

* The notebook contains the full exploratory data analysis, model comparison, hyperparameter tuning, and the code that generated the `.pkl` file.

# **Milestone 1: Data Collection**

Machine learning cannot happen without data. This step is about getting the right dataset before we write a single line of code.

**Activity 1: Download the dataset**

For this project we used startup data sourced from **Crunchbase**, made available on Kaggle. The dataset contains information about over 900 real startups — including their funding history, investor counts, milestones, founding details, geographic location, and whether they were ultimately **acquired** (success) or **closed** (failure).

Dataset name: `startup data.csv`

Download link: [https://www.kaggle.com/datasets/manishkc06/startup-success-prediction](https://www.kaggle.com/datasets/manishkc06/startup-success-prediction)

Place the downloaded file in the root of the project folder (same level as `app.py`).

The target column in this dataset is `status`, which contains two values:
- `acquired` → the startup succeeded (label: **1**)
- `closed` → the startup failed (label: **0**)

# **Milestone 2: Visualizing and Analysing the Data**

Now that the dataset is downloaded, we read it, explore it, and try to understand what's inside before doing any preprocessing or modelling.

**Note: There are many ways to explore a dataset. We used a selection of techniques here. You are free to go deeper and apply more visualization methods on top of what is shown.**

**Activity 1: Importing the Libraries**

We import all the necessary libraries at the top of the notebook. This includes `numpy` and `pandas` for data handling, `matplotlib` and `seaborn` for visualization, `scikit-learn` for model building, and `pickle` for saving the final model. We also silence unnecessary warnings to keep the output clean.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')
import pickle
```

**Activity 2: Read the Dataset**

The dataset is in CSV format, so we use `pd.read_csv()` to load it into a pandas DataFrame. We then check the shape (number of rows and columns), the data types of each column, the column names, and whether any values are missing.

From this initial look, we found the dataset has a mix of numeric features (funding amounts, counts, ages) and categorical ones (state code, category code, status). Several columns also have missing values that we'll need to handle in the preprocessing stage.

**Activity 3: Univariate Analysis**

We start by looking at individual columns to understand how the data is distributed.

**State Distribution (Pie Chart):**
We grouped the `state_code` column — since there are too many unique states — by keeping the top 5 (California, New York, Massachusetts, Texas, Washington) and labelling everything else as "other". A pie chart shows that California alone makes up the largest share by far, which reflects the real-world concentration of startups in Silicon Valley.

**Category Distribution (Pie Chart):**
Similarly, we kept the top startup categories (software, web, mobile, enterprise, advertising, games/video, semiconductor, biotech, hardware, network hosting) and grouped the rest. Software and web startups dominate the dataset.

**Startup Status Distribution (Bar + Pie Chart):**
We plotted both a count bar chart and a pie chart to see how many startups in the dataset were acquired vs closed. This gives us the class distribution of our target variable — an important thing to know before training any model.

**Activity 4: Bivariate Analysis**

We explored how different variables relate to the startup's outcome (success or failure).

**State vs Status (Stacked Bar Chart):**
We plotted each state against the count of acquired vs closed startups. California has the most startups but also a strong success rate. Smaller states have fewer data points, which makes them harder to draw conclusions from.

**State vs Category (Stacked Bar Chart):**
This chart shows which types of startups are most common in each state. California leads across almost all categories, while other states show more concentrated activity in specific sectors.

**Funding Rounds vs Status (Multi-panel Bar Charts):**
We plotted six separate charts — one for each funding type (VC, angel, Round A, B, C, D) — checking whether having gone through that type of funding influenced the startup's final outcome. Startups with VC, angel, and multiple rounds of funding were found to have higher success rates than those without.

**Activity 5: Multivariate Analysis**

We explored relationships involving more than two variables at the same time.

**Category vs Founded Year (Stacked Bar Chart):**
By extracting the founding year from the `founded_at` date column, we plotted how the distribution of startup categories shifted over time. Software and web startups showed strong growth in the early 2000s and peaked around 2007–2008.

**Founded Year vs Average Total Funding (Bar Chart):**
We grouped startups by founding year and calculated the average total funding raised. This revealed that startups founded in certain years attracted significantly more capital — likely reflecting the activity of different funding cycles and economic conditions.

**Activity 6: Descriptive / Statistical Analysis**

We used `data.describe()` to get a statistical summary of all numeric columns — including mean, standard deviation, minimum, maximum, and percentile values. This helps spot potential outliers and understand the scale of different features.

**Correlation Heatmap:**
We computed the correlation matrix for all numeric columns and visualized it using a heatmap. The lower triangle of the heatmap makes it easier to read. From this we could see which features are most correlated with the target and which features are heavily correlated with each other (which can sometimes cause redundancy).

# **Milestone 3: Data Pre-processing**

Now that we understand the data, it's time to clean and prepare it so it's actually suitable for training a machine learning model. Raw data is messy — it has irrelevant columns, missing values, variables with too many categories, and values on wildly different scales. All of that gets addressed here.

**Activity 1: Checking for Null Values**

We used `data.isnull().sum()` to count missing values in each column. Several columns had missing entries. For numeric columns, we filled the missing values with the median of that column — the median is more robust than the mean when outliers are present. After filling with medians, any remaining rows with nulls were dropped entirely.

**Activity 2: Reducing Category Cardinality**

**State Code Variable:**
We first checked whether `state_code` and `state_code.1` contain the same data. They differ by exactly one row (where `state_code.1` has a missing value). So we used `state_code` as the reliable version.

We then confirmed that the top 5 states (CA, NY, MA, TX, WA) account for more than 80% of the data. Everything else was grouped under "other". This reduces noise from rare categories that don't have enough examples to be statistically meaningful.

**Category Code Variable:**
The same logic was applied here — top categories were kept by name; everything else was grouped as "other".

**Activity 3: Dropping Irrelevant Columns**

Many columns in the raw dataset are not useful for prediction. These include identifiers (IDs, names, zip codes), date string columns (we already extracted what we needed), duplicate or near-duplicate columns (like `state_code.1`), and the temporary columns we created during EDA. All of these were dropped in one step.

After dropping, the dataset went from 50+ columns down to a clean, focused set of numeric and binary features that actually carry predictive signal.

**Activity 4: Encoding the Target Variable**

The `status` column contains string values. We converted it to numeric labels:
- `acquired` → **1** (success)
- `closed` → **0** (failure)

This is required because machine learning algorithms work with numbers, not strings.

**Activity 5: Scaling the Data**

Features in this dataset are on very different scales — for example, `funding_total_usd` can be in the millions while `milestones` is typically between 0 and 10. Without scaling, distance-based methods and gradient methods would be biased toward features with larger values.

We applied `StandardScaler` from scikit-learn, which transforms each feature to have a mean of 0 and a standard deviation of 1. The scaler was fit **only on the training data** and then applied to both train and test sets — this is important to avoid data leakage.

**Activity 6: Splitting Data into Train and Test**

We separated the features (`X`) from the target (`y`) and then split the data into 70% training and 30% testing using `train_test_split()` from scikit-learn. We used `stratify=y` to ensure that both sets have the same ratio of successful to failed startups, and `random_state=116` to make the results reproducible.

# **Milestone 4: Model Building**

With clean, scaled data in hand, it's time to build and evaluate models. We tried four different classification algorithms, compared their performance, tuned the best ones, and then identified the optimal set of features before saving the final model.

**Activity 1: Logistic Regression**

Logistic Regression is our baseline model. It's simple, interpretable, and works well as a first attempt on classification tasks. We initialized it with `max_iter=1000` to ensure convergence and evaluated it using a reusable `evaluate_model()` function that prints accuracy, precision, recall, F1-score, and a confusion matrix heatmap.

**Activity 2: Support Vector Machine (SVM)**

We trained an SVM with an RBF (radial basis function) kernel. SVMs are known to work well in high-dimensional spaces and tend to generalize well when the data is not too noisy. The RBF kernel allows the decision boundary to be non-linear, which is helpful when classes are not linearly separable.

**Activity 3: Decision Tree**

We trained a Decision Tree classifier with default parameters. Decision Trees are highly interpretable — you can literally follow the tree's branches to understand why a prediction was made. However, they tend to overfit if not constrained, which should be visible in the gap between training and test accuracy.

**Activity 4: Random Forest**

Random Forest builds many decision trees on random subsets of the data and features and then combines their votes. This ensemble approach significantly reduces overfitting compared to a single tree. We started with 100 trees (`n_estimators=100`) and expected this model to outperform the others.

**Activity 5: Comparing the Models (Before Tuning)**

After training all four models, we put their results into a DataFrame and compared them side by side with a multi-bar chart (train accuracy, test accuracy, precision, recall, F1-score for each model). This gave us an honest look at which algorithm performs best before we do any tuning.

Random Forest came out ahead on test accuracy, but we still ran the tuning process on all four models for completeness.

**Activity 6: Hyperparameter Tuning with GridSearchCV**

We used `GridSearchCV` with 5-fold cross-validation to search for the best hyperparameters for each model:

- **Logistic Regression:** Tuned `C`, `penalty`, and `solver`.
- **SVM:** Tuned `C`, `kernel`, and `gamma`.
- **Decision Tree:** Tuned `max_depth`, `min_samples_split`, `min_samples_leaf`, and `criterion`.
- **Random Forest:** Tuned `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`.

After tuning, we compared the before vs after test accuracies to see whether it made a meaningful difference. A side-by-side bar chart with accuracy values annotated on top of each bar was generated to make this easy to read.

**Activity 7: Feature Importance and Optimal Feature Selection**

Using the best Random Forest model from GridSearchCV, we extracted the importance score of every feature and ranked them in a horizontal bar chart. The most important features turned out to be things like:

- Number of investor/advisor relationships
- Age at last milestone
- Total funding raised (USD)
- Number of milestones achieved
- Age at first funding

We then ran an experiment: starting from the single most important feature, we kept adding the next most important feature one by one and trained a fresh Random Forest at each step. This gave us an "Accuracy vs Number of Features" line plot, from which we identified the **optimal number of features** — the point where adding more features stops helping (or even slightly hurts).

**Activity 8: Final Model and Saving**

We retrained the best Random Forest on only the optimal set of features. This final model achieved a **test accuracy of 79.06%** with a 5-fold stratified cross-validation approach. We then saved the model as a pickle file (`random_forest_model.pkl`), bundling three things together:

- The trained `RandomForestClassifier`
- The fitted `StandardScaler`
- The list of optimal features (so the Flask backend knows which inputs to expect)

```python
model_data = {
    'model': final_model,
    'scaler': scaler,
    'features': optimal_features,
    'all_features': X.columns.tolist()
}
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)
```

| Metric | Value |
|--------|-------|
| Algorithm | Random Forest Classifier |
| Test Accuracy | **79.06%** |
| CV Strategy | 5-Fold Stratified |
| Optimal Features | 23 selected from full feature set |

# **Milestone 5: Application Building**

With the model saved, we built a complete web application around it so that any user — even one with zero knowledge of machine learning — can enter their startup's details and get a prediction with a confidence score.

**Activity 1: Building the HTML Page**

We created a single `index.html` file inside the `templates/` folder. The page is designed with a dark glassmorphism aesthetic — a frosted-glass card layout on a animated gradient background. Key features of the UI include:

- **Input form** with labeled fields for all the model inputs (funding amount, number of rounds, number of investors, milestones, age at funding, etc.)
- **Hover tooltips** on every single field — so users know exactly what to enter without having to look anything up.
- **INR Converter panel** — since the dataset is in USD, Indian founders can enter a value in rupees and get the USD equivalent instantly with a live ₹ lakh / crore breakdown.
- **City Quick-Fill buttons** — clicking any city chip (Bengaluru, Mumbai, Delhi, Hyderabad, Chennai, Kolkata, San Francisco, New York, etc.) automatically fills in that city's coordinates.
- **Collapsible Glossary** at the bottom of the page — a plain-English explanation of every term used in the form.
- **Prediction result card** that appears after submission with a large SUCCESS or FAILURE label, a confidence percentage, and a short explanatory message.

**Activity 2: Build the Python Code (Flask Backend)**

`app.py` is the brain of the web application. Here's what it does:

**Importing libraries and loading the model:**

```python
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

with open('random_forest_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']
all_features = model_data['all_features']
```

**Rendering the home page:**
The `/` route simply renders `index.html` when a user visits the app in their browser.

**The `/predict` endpoint:**
This is a POST route that:
1. Receives the form data as JSON from the frontend.
2. Builds an input array in the exact column order the scaler expects.
3. Scales the input using the pre-fitted `StandardScaler`.
4. Extracts only the optimal features the model was trained on.
5. Runs `model.predict()` and `model.predict_proba()`.
6. Returns a JSON response with the prediction label, confidence score, and a human-readable message.

```python
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_array = np.array([[data.get(f, 0) for f in all_features]])
    input_scaled = scaler.transform(input_array)
    feature_indices = [all_features.index(f) for f in features if f in all_features]
    input_final = input_scaled[:, feature_indices]
    prediction = model.predict(input_final)[0]
    probability = model.predict_proba(input_final)[0]
    ...
```

**Activity 3: Run the Application**

To run the project locally:

1. Open your terminal and navigate to the project folder:
   ```
   cd path/to/startup_pred
   ```

2. Install all dependencies (if not already done):
   ```
   pip install -r requirements.txt
   ```

3. Start the Flask server:
   ```
   python app.py
   ```

4. Open your browser and go to:
   ```
   http://localhost:5000
   ```

5. Fill in the form fields, click **Predict**, and the model will return a **SUCCESS** or **FAILURE** verdict along with a confidence percentage.

The application runs entirely on your local machine in debug mode. No cloud setup, no Docker, no environment variables needed.
