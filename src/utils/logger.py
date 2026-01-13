import logging 
import os 
from datetime import datetime 

LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE= os.path.join(LOG_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig (filename=LOG_FILE, 
                     format="[%(asctime)s] %(levelname)s - %(message)s",
                     level= logging.INFO)


logger= logging.getLogger()