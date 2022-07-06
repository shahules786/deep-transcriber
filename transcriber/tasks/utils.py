
import logging
import os
import json
from random import Random
import zipfile
import numpy as np
from huggingface_hub import cached_download, hf_hub_url
import torch


def min_value_check(arg,value,name):
    if arg<value:
        raise ValueError(f"{str(name)} should be greater than or equal to {value}")
        
    return arg 
    
def path_check(path):

    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exists")
    return path

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

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def hertz_to_mel(x):
    return 2595 * np.log10(1 + x / 700)

def mel_to_hertz(x):
    return 700 * (10 ** (x / 2595) - 1)

def random_generation():
    
    rng = Random()
    global_seed = int(os.environ.get("SEED","1000"))
    worker_info = torch.utils.data.get_worker_info()

    if worker_info is None:
        num_workers = 1
        worker_id = 0
    else:
        num_workers = worker_info.num_workers
        worker_id = worker_info.id

    rng.seed(global_seed*worker_id+num_workers)
    return rng


class EarlyStopping:

    def __init_(
        self,
        patience:int=10,
        mode="min",
        filename="model.pth",
        directory="./model"

    ):
        self.patience = patience 
        self.counter = 0
        if mode in ("min","max"):
            self.mode = mode
        else:
            raise ValueError("mode must be either 'min' or 'max'")
        self.best_loss = None
        self.early_stop = False
        if not os.path.exists(directory):
            logging.info("CREATING DIR TO SAVE MODEL...")
            os.mkdir(directory)

        self.filepath = os.path.join(directory,filename)

        



    def __call__(
        self,
        loss,
        model

    ):
        if self.best_loss == None:
            self.best_loss = loss
        else:
            if self.mode == "min":
                if loss < self.best_loss:
                    self.best_loss = loss
                    self.save_checkpoint(model)
                else:
                    self.counter += 1
            else:
                if loss > self.best_loss:
                    self.best_loss = loss
                    self.save_checkpoint(model)
                else:
                    self.counter += 1


        if self.counter >= self.patience:
            print(f"Early stopping with best loss {self.best_loss}")
            self.early_stop = True

    def save_checkpoint(
        self,
        model
    ):
        torch.save(model.state_dict(),self.filepath)





    

