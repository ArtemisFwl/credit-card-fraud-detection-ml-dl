

from src.utils.logger import logger
from src.utils.exceptions import FraudException
#___________________________________________________________________________________

from src.data.data_loader import DataLoader

if __name__== "__main__":
  try:
    logger.info("Pipeline Started")

    loader=DataLoader("data/creditcard.csv")

    df=loader.load_data()
    logger.info("Dataloader Setup completed successfully")

  except Exception as e:
    logger.error("Pipeline Failed")
    raise FraudException("Main Pipeline Failed")
#___________________________________________________________________________________
from src.preprocessing.preprocessor  import Preprocessor

preprocessor=Preprocessor()

X,y= preprocessor.split_features_target(df)
X_train, X_test, y_train, y_test =preprocessor.train_test_split_data(X, y)
logger.info("Preprocessing step completed successfully")

#___________________________________________________________________________________
from src.models.logistic_regression import LogisticRegressionTrainer
trainer= LogisticRegressionTrainer()

model=trainer.train(X_train, y_train)

report, auc=trainer.evaluate(X_test, y_test)
logger.info("Baseline Logistic Regression COmpleted")

#___________________________________________________________________________________
from src.models.logistic_regression import LogisticRegressionTrainer
trainer= LogisticRegressionTrainer()

model=trainer.train(X_train, y_train)

report, auc=trainer.evaluate(X_test, y_test)
logger.info("Baseline Logistic Regression COmpleted")

#___________________________________________________________________________________
from src.models.random_forest import RandomForestTrainer
rf_model = RandomForestTrainer()
rf_model.train(X_train, y_train)
rf_report, rf_auc = rf_model.evaluate(X_test, y_test)

logger.info("Baseline random forest Regression COmpleted")


import numpy as np
from sklearn.metrics import precision_recall_curve

# ===== Threshold Tuning (Random Forest) =====
y_proba_rf = rf_model.model.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, y_proba_rf)

# Simple inspection
for t, p, r in zip(thresholds[::50], precision[::50], recall[::50]):
    logger.info(f"Threshold={t:.3f} | Precision={p:.3f} | Recall={r:.3f}")

# Pick threshold where recall >= 0.90
target_recall = 0.90
best_threshold = 0.5

for t, r in zip(thresholds, recall[1:]):
    if r >= target_recall:
        best_threshold = t
        break

logger.info(f"Chosen Threshold based on recall>=0.90: {best_threshold}")

y_pred_custom = (y_proba_rf >= best_threshold).astype(int)

from sklearn.metrics import classification_report
logger.info("Random Forest with Custom Threshold Report:")
logger.info(classification_report(y_test, y_pred_custom))

#____________________________________________________________________

from sklearn.metrics import precision_recall_curve
import numpy as np

y_proba_lr = lr_model.model.predict_proba(X_test)[:, 1]

precision_lr, recall_lr, thresholds_lr = precision_recall_curve(
    y_test, y_proba_lr
)

for t, p, r in zip(thresholds_lr[::50], precision_lr[::50], recall_lr[::50]):
    logger.info(f"[LR] Threshold={t:.3f} | Precision={p:.3f} | Recall={r:.3f}")


target_recall = 0.95
best_lr_threshold = 0.5

for t, r in zip(thresholds_lr, recall_lr[1:]):
    if r >= target_recall:
        best_lr_threshold = t
        break

logger.info(f"[LR] Chosen Threshold (recall>=0.95): {best_lr_threshold}")

y_pred_lr_custom = (y_proba_lr >= best_lr_threshold).astype(int)

logger.info("Logistic Regression with Custom Threshold Report:")
logger.info(classification_report(y_test, y_pred_lr_custom))

