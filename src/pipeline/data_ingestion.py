from src.common.lib import *
from src.common.dir import *
from src.common.var import *
from src.utils.utils import *
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

STAGE_NAME = "Data Ingestion"

class DataIngestion:
    def __init__(self):
        self.train_idx = []
        self.test_idx = []
        self.dataset = torch.load(os.path.join(ARGS_CONFIG_DIR, ARGS_CONFIG_NAME), weights_only=False)['dataset']

    def data_ingestion(self):
        print(self.dataset)
        if self.dataset == "cnh":
            df = read_csv(INPUT_CSV_FILE)
            df = df.drop(columns=['patient_id'])
            ## convert the string to integer 
            mapping_dict = {'Positive': 1, 'Negative': 0}
            df['label'] = df['label'].map(mapping_dict)
            ## save the file to csv
            write_csv(df, UPDATED_CSV_FILE)
            return df
        
        else:
            positive_dir_path = Path(os.path.join(KAGGLE_IMG_DIR,'train/phar'))
            pos_img_names = []

            for file_path in positive_dir_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif']:
                    pos_img_names.append(os.path.join("train/phar",file_path.name)) 


            negative_dir_path = Path(os.path.join(KAGGLE_IMG_DIR,'train/no'))
            neg_img_names = []

            for file_path in negative_dir_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif']:
                    neg_img_names.append(os.path.join("train/no",file_path.name)) 

            df_pos = pd.DataFrame(pos_img_names, columns=['ImageName'])
            df_pos.reset_index(drop=True, inplace=True)
            df_pos['Labels'] = 1
            df_neg = pd.DataFrame(neg_img_names, columns=['ImageName'])
            df_pos.reset_index(drop=True, inplace=True)
            df_neg['Labels'] = 0

            df = pd.concat([df_pos, df_neg])
            write_csv(df, KAGGLE_CSV_FILE)

    def train_test_split(self):
        ## split train test dataset in stratified way. Train and test dataset have equal proportion of positives and negatives
        if self.dataset == "cnh":
            df = read_csv(UPDATED_CSV_FILE)
        else:
            df = read_csv(KAGGLE_CSV_FILE)
        labels = df.iloc[:, 1]
        indices = range(len(df))
        train_idx, temp_idx = train_test_split(
            indices, 
            test_size=0.3, 
            stratify=labels, 
            random_state=42
        )

        val_idx, test_idx = train_test_split(
            temp_idx, 
            test_size=0.5, 
            stratify=labels.iloc[temp_idx], 
            random_state=42
        )
        
        print("Train dataset value counts: ",labels[train_idx].value_counts())
        print("Test dataset value counts: ",labels[test_idx].value_counts())
        print("val dataset value counts: ",labels[test_idx].value_counts())

        return train_idx, test_idx, val_idx

if __name__ == "__main__":
    obj = DataIngestion()
    obj.data_ingestion()
    obj.train_test_split()
