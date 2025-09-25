from pathlib import Path
import glob
import pickle
import pandas as pd
import json


class RawDataReader():
    def __init__(self, raw_data_directory: Path):
        self.raw_data_directory = raw_data_directory

    def _read(self, fname:str)->pd.DataFrame:
        return pd.read_csv(fname)

    def read_ir_triplets(self) -> dict:
        with open(self.raw_data_directory / "ir_triplets" / "ir_triplets.json", "r") as f:
            data = json.load(f)
        return  data

    def read_deer(self) -> pd.DataFrame:
        deer_data = pd.concat([pd.read_excel(self.raw_data_directory / "deer" / 'Hypothetical_Induction_test.xlsx'),
                                pd.read_excel(self.raw_data_directory / "deer" / 'Hypothetical_Induction_train.xlsx'),
                                pd.read_excel(self.raw_data_directory / "deer" / 'Hypothetical_Induction_val.xlsx')])
        return deer_data



class PreprocessedDataReader():
    def __init__(self, preprocessed_data_directory: Path):
        self.preprocessed_data_directory = preprocessed_data_directory

    def _read(self, fname:str)->pd.DataFrame:
        return pd.read_csv(fname)

