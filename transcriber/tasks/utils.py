
import logging
import os
import json
import zipfile
from huggingface_hub import cached_download, hf_hub_url



def min_value_check(arg,value):
    if arg<=value:
        raise ValueError(f"{str(arg)} should be greater than or equal to {value}")
        
    return True
    
def path_check(path):

    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exists")
    return True

def download_data_kaggle(file_name='programmerrdai/house-price-to-the-moon',save_path:str="./data/"):

    credentials = json.load(open("./data/kaggle.json","r"))
    os.environ['KAGGLE_USERNAME'] = credentials["username"]
    os.environ['KAGGLE_KEY'] = credentials["key"]
    
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(file_name, path=save_path)
    logging.info("Data downloaded..")
    with zipfile.ZipFile(os.path.join(save_path,f'{file_name.split("/")[-1]}.zip'), 'r') as zip_ref:
        zip_ref.extractall(save_path)

def load_file_hf(filename, model_id, revision_id):
    url = hf_hub_url(model_id=model_id, revision_id=revision_id,filename=filename)
    path = cached_download(
                url=url,
            )
    return path


if __name__ == "__main__":
    download_data_kaggle()