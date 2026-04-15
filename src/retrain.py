import os
import logging
from src.train import train

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def trigger_retraining():
    """
    Simulate a retraining trigger. 
    In a real scenario, this might check for data drift or be scheduled.
    """
    logger.info("Retraining trigger received.")
    try:
        train()
        logger.info("Retraining completed successfully.")
        return True
    except Exception as e:
        logger.error(f"Retraining failed: {e}")
        return False

if __name__ == "__main__":
    trigger_retraining()
