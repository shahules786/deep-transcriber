import os
import torch
import json

from transcriber.tasks.embeddings.model import Embeder
from transcriber.tasks.utils import load_file_hf

class PretrainedEmbeder:

    def __init__(
        self,
        pretrained_model_name_or_path:str,
        device:str="cpu"
    ):
        if os.path.isfile(pretrained_model_name_or_path):
            config = os.path.join(pretrained_model_name_or_path,'config.json')
            model = os.path.join(pretrained_model_name_or_path,'pytorch.pt')
        else:
            if "@" in pretrained_model_name_or_path:
                model_id = pretrained_model_name_or_path.split('@')[0]
                revision_id = pretrained_model_name_or_path.split('@')[1]
            config = load_file_hf("config.json",model_id,revision_id)
            model = load_file_hf("pytorch.pt",model_id,revision_id)
            
        self.device = device ##change
        self.model = self._load_model(model,config)
         
    def _load_model(self,model_path, config_path):

        config = json.load(open(config_path))
        model = Embeder(**config)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model

    def preprocess(
        self,
    ):
        pass

    def predict(
        self,
    ):
        pass

