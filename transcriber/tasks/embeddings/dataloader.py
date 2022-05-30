import logging
import torch
from torch.utils.data import DataLoader,Dataset
import numpy as np
import glob 
import os 
import random



class TimitDataset(Dataset):

    def __init__(
        self,
        directory:str,
        n_speakers:int,
        n_utterances:int,
        return_tensors=False,
    ):
        self.directory = directory
        self.n_utterances = n_utterances
        self.n_speakers = n_speakers
        self.utterances = self.filter_utterances()
        self.return_tensors = return_tensors


    def filter_utterances(
        self,
    ):
        selected_files = []
        np_files = [file for file in glob.glob(os.path.join(self.directory,"*")) if file.endswith("npy")]
        for file in np_files:
            if np.load(file).shape[0] >= self.n_utterances:
                selected_files.append(file)

        return selected_files


    def __getitem__(
        self, 
        index
    ):

        npy_file = np.load(self.utterances[index])
        utter_start = np.random.randint(0,npy_file.shape[0] - self.n_utterances + 1)
        utterances = npy_file[utter_start:utter_start+self.n_utterances]
        
        if not self.return_tensors:
            return utterances.transpose(0,2,1)
        else:
            return torch.from_numpy(utterances.transpose(0,2,1))
    def __len__(
        self,

    ):
        return len(self.utterances)


class TimitCollate:

    def __init__(
        self,
        n_speakers:int,
        n_utterances:int
    ):
        self.n_speakers = n_speakers
        self.n_utterances = n_utterances


    def __call__(
        self,
        batch
    ):
        batch = np.array(batch)
        batch = batch.reshape((self.n_speakers*self.n_utterances, batch.shape[2],batch.shape[3]))
        permute = random.sample(range(0,self.n_speakers*self.n_utterances),self.n_speakers*self.n_utterances)
        unpermute = permute.copy()
        for i,j in enumerate(permute):
            unpermute[j] = i

        return {"data": torch.from_numpy(batch[permute]),
                "unpermute":unpermute
                }




