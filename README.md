# Credit Card Fraud Detection (ML + DL)

End-to-end Machine Learning project to detect fraudulent credit card transactions using
Logistic Regression, Random Forest, and XGBoost with threshold tuning.

## Dataset Setup
The dataset is intentionally excluded from the repository.
Please follow the instructions in `data/README.md` to download and place the dataset locally.
---

## 🚀 Project Highlights
- Clean modular project structure
- Custom logging & exception handling
- Multiple models comparison (ROC-AUC)
- Threshold tuning for recall-critical use case
- Production-style pipeline mindset

---

## 📂 Project Structure
credit-card-fraud-detection-ml-dl/
│
├── src/
│ ├── data/
│ │ └── data_loader.py
│ ├── preprocessing/
│ │ ├── preprocessor.py
│ │ └── pipeline.py
│ ├── models/
│ │ ├── logistic_regression.py
│ │ ├── random_forest.py
│ │ └── xgboost_model.py
│ └── utils/
│ ├── logger.py
│ └── exceptions.py
│
├── artifacts/ # saved models (ignored in git)
├── data/ # dataset (ignored in git)
├── main.py # training + evaluation runner
├── requirements.txt
└── README.md


---

## 🧠 Models Implemented
| Model | ROC-AUC |
|------|--------|
| Logistic Regression | ~0.97 |
| Random Forest | ~0.95 |
| XGBoost | ~0.97 |

---

## 🎯 Why Threshold Tuning?
Fraud detection is **recall-critical**.
Missing fraud is costlier than false alarms.

Custom probability thresholds were tuned to:
- Maintain recall ≥ 90%
- Compare precision-recall tradeoff

---

## 🛠️ How to Run

### 1️⃣ Create virtual environment
```bash
python -m venv ccfd_venv
ccfd_venv\Scripts\activate

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run pipeline
python main.py

📊 Logs

All steps are logged with timestamps using custom logger:

logs/

🔮 Future Improvements

SMOTE / class imbalance handling

Hyperparameter tuning (GridSearch / Optuna)

Model registry

FastAPI inference API

Deep Learning model (ANN)

👨‍💻 Author

Aman Deep
AI / ML Engineer
