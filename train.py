from src.pipeline.data_ingestion import DataIngestion
from src.pipeline.create_dataloader import CreateDataloader
from src.pipeline.train_model import ModelTraining
from src.common.dir import *
from src.utils.utils import create_directory
from src.common.lib import *
from src.common.var import *


import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

import argparse
parser = argparse.ArgumentParser(description="Script parameters")

# 2. Add arguments
parser.add_argument("--dataset", type=str, default="cnh", help="Choose the dataset")

# 3. Parse the arguments
args = parser.parse_args()

eval_config = {
    'dataset': args.dataset,
}
create_directory(ARGS_CONFIG_DIR)
torch.save(eval_config, os.path.join(ARGS_CONFIG_DIR,ARGS_CONFIG_NAME))

STAGE_NAME = "Data Ingestion"
try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestion()
    data_ingestion.data_ingestion()
    train_idx, test_dx, val_idx = data_ingestion.train_test_split()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx===========x")
except Exception as e:
    logging.exception(e)
    raise e

STAGE_NAME = "Prepare Data Loader"
try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    create_dataloader = CreateDataloader(train_idx, test_dx, val_idx)
    train_loader = create_dataloader.create_train_dataloader(TRAIN_BATCH_SIZE)
    val_loader = create_dataloader.create_val_dataloader(VAL_BATCH_SIZE)
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx===========x")
except Exception as e:
    logging.exception(e)
    raise e

STAGE_NAME = "Train Model"
try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    modeltraining = ModelTraining(train_loader, val_loader)
    modeltraining.model_train()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx===========x")
except Exception as e:
    logging.exception(e)
    raise e

