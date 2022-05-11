
import os
import torch

from transcriber.tasks.utils import min_value_check, path_check

class EmbedTrainer:

    def __init__(
        self,
        train:str,
        test:str,
        num_speakers:int,
        num_utters:int,
        batch_size:int,
        epochs:int,
        lr:float,
        hidden_layers:int,
        num_layers:int,
        embedding_dim:int,
        device:str="cpu"
    ):

        if path_check(train):
            self.train = train

        if path_check(test):
            self.test = test

        if min_value_check(num_speakers,0):
            self.num_speakers = num_speakers

        if min_value_check(num_utters,0):
            self.num_layers = num_layers

        if min_value_check(batch_size,0):
            self.num_layers = num_layers

        if min_value_check(epochs,0):
            self.num_layers = num_layers

        if min_value_check(hidden_layers,0):
            self.num_layers = num_layers

        if min_value_check(embedding_dim,0):
            self.num_layers = num_layers

        if device not in ("cpu","cuda"):
            raise ValueError("device should be cpu or cuda")
        else:
            if getattr(torch,device).is_available():
                self.device = torch.device(device)
            else:
                raise ValueError(f"{device} not available!")
        
    
    def train(
        self,
    ):
        pass


    def eval(
        self,

    ):
        pass





if __name__ == "__main__":
    import yaml

    with open('conf.yaml') as file:
        args = yaml.full_load(file)

    trainer = EmbedTrainer(**args)  ##change this to pass keyword arguments 
    trainer.train()




