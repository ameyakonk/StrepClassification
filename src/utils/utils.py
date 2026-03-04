from src.common.lib import *
from src.common.dir import *
from src.common.var import *
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

def read_csv(csv_name):
    df = pd.read_csv(os.path.join(CSV_DIR,csv_name))
    return df

def write_csv(df, csv_name):
    df.to_csv(os.path.join(CSV_DIR,csv_name), index = False)

def create_directory(dir_path):
    try:
        os.mkdir(dir_path)
        logging.info(f"Directory '{dir_path}' created.")
    except FileExistsError:
        logging.info(f"Directory '{dir_path}' already exists.")
    except OSError as e:
        logging.error(f"Error creating directory: {e}")