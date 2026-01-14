from src.data.data_loader import DataLoader

from src.utils.logger import logger
from src.utils.exceptions import FraudException

if __name__== "__main__":
  try:
    logger.info("Pipeline Started")

    loader=DataLoader("data/creditcard.csv")

    df=loader.load_data()
    logger.info("Dataloader Setup completed successfully")

  except Exception as e:
    logger.error("Pipeline Failed")
    raise FraudException("Main Pipeline Failed")
  
from src.preprocessing.preprocessor  import Preprocessor

preprocessor=Preprocessor()

X,y= preprocessor.split_features_target(df)

X_train, X_test, y_train, y_test =preprocessor.train_test_split_data(X, y)

logger.info("Preprocessing step completed successfully")