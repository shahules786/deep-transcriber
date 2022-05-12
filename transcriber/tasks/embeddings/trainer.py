
import os
import torch
from torch.utils.data import DataLoader

from transcriber.tasks.embeddings.dataloader import TimitDataset,TimitCollate
from transcriber.tasks.utils import min_value_check, path_check

class EmbedTrainer:

    def __init__(
        self,
        train:str,
        test:str,
        n_speakers:int,
        n_utterances:int,
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

        if min_value_check(n_speakers,0):
            self.n_speakers = n_speakers

        if min_value_check(n_utterances,0):
            self.n_utterances = n_utterances

        if min_value_check(batch_size,0):
            self.batch_size = batch_size

        if min_value_check(epochs,0):
            self.epochs = epochs
        
        if min_value_check(num_layers,0):
            self.num_layers = num_layers

        if min_value_check(hidden_layers,0):
            self.hidden_layers = hidden_layers

        if min_value_check(embedding_dim,0):
            self.num_layers = num_layers

        if device not in ("cpu","cuda"):
            raise ValueError("device should be cpu or cuda")
        else:
            if getattr(torch,device).is_available():
                self.device = torch.device(device)
            else:
                raise ValueError(f"{device} not available!")
        self.lr = lr
    
    def train(
        self,
    ):
        datalaoders = self._prepare_dataloaders()
        pass
        
    def _prepare_dataloaders(
        self,
    ):
        train_dataset = TimitDataset(directory=self.train, n_utterances=self.n_utterances,
                                        n_speakers = self.n_speakers)
        collate_fn = TimitCollate(n_speakers = self.n_speakers, n_utterances=self.n_utterances)
        train_dataset = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)

        valid_dataset = TimitDataset(directory=self.train, n_utterances=self.n_utterances,
                                        n_speakers = self.n_speakers)
        collate_fn = TimitCollate(n_speakers = self.n_speakers, n_utterances=self.n_utterances)
        valid_dataset = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)

        return {"train":train_dataset,
                "valid":valid_dataset}

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




