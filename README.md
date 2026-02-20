# 🚀 Prosperity Prognosticator — Startup Success Predictor

> *Will your startup make it? Let the machine decide.*

A machine-learning web app that predicts whether a startup will succeed or fail — built with a Random Forest classifier trained on real-world Crunchbase startup data, served through a clean Flask backend, and wrapped in a glassmorphism UI designed to be understood by anyone, not just data scientists.

---

## 🎯 What does it actually do?

You fill in a short form about your startup — how long it took to get funded, how many investors you have, how much money you've raised, where you're based — and the model gives you a **SUCCESS / FAILURE prediction** with a confidence score.

No jargon. Every field has a plain-English tooltip explaining exactly what to enter. Indian founders get an INR conversion table and city quick-fill buttons.

---
<img width="1874" height="940" alt="image" src="https://github.com/user-attachments/assets/748f7b0d-cf61-49d0-9441-b4730f040d33" />
<img width="1201" height="818" alt="image" src="https://github.com/user-attachments/assets/22d911bf-98f5-4d79-b11b-8abe29a3c1db" />

<img width="1868" height="934" alt="image" src="https://github.com/user-attachments/assets/4df274e7-eff4-4ea6-9b69-39efaa493f9c" />

<img width="1487" height="929" alt="image" src="https://github.com/user-attachments/assets/9f43aab7-376e-46df-8475-c3dd1d4f7372" />




## ✨ Features

- **79% Accuracy** — Random Forest trained with GridSearchCV on 900+ real startups
- **Live INR Converter** — enter funding in USD, see ₹ Lakh / Crore instantly
- **City Quick-Fill** — one click fills coordinates for SF, NY, Boston, Austin, Bengaluru, Mumbai, Delhi, Hyderabad, Chennai, Kolkata, Ahmedabad, Pune, and more
- **Hover Tooltips** on every field — so users understand "Number of Relationships" without Googling it
- **Collapsible Glossary** — plain-English guide to every term used in the model
- **Glassmorphism UI** — smooth animated background, Inter font, dark theme

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Model | scikit-learn 1.6.1 — Random Forest Classifier |
| Backend | Python 3.13 + Flask |
| Frontend | Pure HTML / CSS / Vanilla JS |
| Dataset | [Kaggle — Startup Success Prediction](https://www.kaggle.com/datasets/manishkc06/startup-success-prediction) |

---

## 📂 Project Structure

```
startup_pred/
├── templates/
│   └── index.html              ← The web UI (tooltips, INR, city picker)
├── app.py                      ← Flask server + /predict API
├── random_forest_model.pkl     ← Trained model (scaler + features bundled)
├── startup data.csv            ← Raw Crunchbase dataset
├── startup-prediction-eda-model.ipynb  ← EDA + model training walkthrough
└── requirements.txt
```

---

## ⚡ Running Locally

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Start the server**
```bash
python app.py
```

**3. Open your browser**
```
http://localhost:5000
```

That's it. No Docker, no config files, no environment variables.

---

## 🤖 How the Model Works

The dataset contains startup metadata from Crunchbase — funding amounts, number of rounds, investor counts, milestones, and location. The target label is whether a startup was **acquired** (success) or **closed** (failure).

**Training pipeline:**
1. Drop irrelevant columns (names, dates, IDs)
2. Encode categorical features (state, category)
3. Fill missing values with column medians
4. Scale features with `StandardScaler`
5. `GridSearchCV` over Random Forest hyperparameters (5-fold CV)
6. Feature importance ranking → select optimal feature subset
7. Final model retrained on best features → **79.06% test accuracy**

**Key features the model cares most about:**
- Number of investor/advisor relationships
- Age at last milestone
- Total funding raised (USD)
- Number of milestones achieved
- Age at first funding

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Test Accuracy | **79.06%** |
| CV Strategy | 5-Fold Stratified |
| Optimal Features | 23 selected from full feature set |

---

## 🌍 Note for Indian Founders

The training data is US-centric (Crunchbase, circa 2013). For Indian startups:
- Use the **INR reference table** on the funding page to convert your raise amount to USD
- Click a **city chip** (Bengaluru, Mumbai, etc.) to auto-fill coordinates
- Set the **India / Other** toggle ON (it's on by default)
- The model will still give a meaningful signal — just treat the confidence score as directional, not absolute

---

## 📄 License

MIT — do whatever you want with it.

---

*Built with curiosity and way too much time spent on CSS transitions.*

