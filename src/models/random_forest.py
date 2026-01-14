from sklearn.ensemble import RandomForestClassifier 

from sklearn.metrics import classification_report, roc_auc_score

from src.utils.logger import logger

from src.utils.exceptions import FraudException

class RandomForestTrainer:
  def __init__(self):
    self.model=RandomForestClassifier(
      n_estimators= 100,
      max_depth=None,
      class_weight="balanced",
      n_jobs=-1,
      random_state=42
    )

  def train(self, X_train, y_train):
    try: 
      logger.info("Training Random Forest Model")
      self.model.fit(X_train, y_train)
      logger.info("Random Forest Training Completed")
      return self.model
    
    except Exception as e:
      logger.error("Random Forest training Failed")
      raise FraudException("Random Forest Training Failed",e)
    
  def evaluate(self, X_test, y_test):
    try:
      logger.info("Evaluate Random Forest model")

      y_pred=self.model.predict(X_test)
      y_proba=self.model.predict_proba(X_test)[:,1]

      report=classification_report(y_test,y_pred)
      auc=roc_auc_score(y_test, y_proba)

      logger.info(f"Random Forest Classification Report:\n{report}")
      logger.info(f"Random Forest ROC-AUC score: {auc}")

      return report, auc
    
    except Exception as e:
      logger.error("Random Forest Evaluation Failed")
      raise FraudException("Random Forest Evaluation Failed", e)
