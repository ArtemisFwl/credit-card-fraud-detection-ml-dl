from sklearn.model_selection import train_test_split
from src.utils.logger import logger
from src.utils.exceptions import FraudException


class Preprocessor:

  def split_features_target(self, df):
    try:
      logger.info("Splitting Features and Data")

      X=df.drop("Class", axis=1)
      y=df["Class"]

      logger.info(f"Features Shape:{X.shape}, Target shape: {y.shape}")
      return X, y
    
    except Exception as e:
      logger.error("Feature target Split Failed")
      raise FraudException("Feature Target Split Failed", e)
    
  def train_test_split_data (Self, X,y, test_size=0.2, random_state=42):
    try: 
      logger.info("Performing train test split")

      X_train, X_test, y_train, y_test =train_test_split(
        X, 
        y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
      )

      logger.info("fTrain Shape: {X_train.shape}, Test Shape: {X_test.shape}")

      return X_train, X_test, y_train, y_test
    
    except Exception as e:
      logger.error("Train Test Split Failed")
      raise FraudException("Train test split failed",e)