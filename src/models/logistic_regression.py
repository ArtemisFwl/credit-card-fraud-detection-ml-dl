from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report , roc_auc_score

from src.utils.logger import logger
from src.utils.exceptions import FraudException

class LogisticRegressionTrainer:
  def __init__(self):
    self.model=LogisticRegression(
      max_iter=1000,
      class_weight='balanced'
    )

  def train(self, X_train, y_train):
    try:
      logger.info("training Logistic regression Model")
      self.model.fit(X_train, y_train)
      logger.info("Model training Completed")
      return self.model 
    except Exception as e:
      logger.error("Model training Failed")
      raise FraudException("Logistic Regression Training Failed",e)
  
  def evaluate(self, X_test, y_test):
    try:
      logger.info("Evaluating the Logistic Regression model")
      y_pred=self.model.predict(X_test)
      y_proba=self.model.predict_proba(X_test)[:,1]

      report =classification_report(y_test, y_pred)
      auc=roc_auc_score(y_test, y_proba)

      logger.info(f"Classification report: \n {report}")
      logger.info(f"ROC-AUC Score: {auc}")

      return report, auc
    
    except Exception as e:
      logger.error("Model Evaluation Failed ")
      raise FraudException("Logistic Regression Evaluation Failed")
