import itertools
from random import sample
from torch.utils.data import IterableDataset
import numpy as np
import librosa
import torch
from pyannote.core import Segment
import math

from transcriber.tasks.utils import softmax,random_generation




class AMIDataset(IterableDataset):

    def __init__(
        self,
        protocol,
        duration=5,
        sampling_rate=None,
        phase="train"
    ):

        self.sampling_rate = sampling_rate
        self.duration = duration
        self.data=[]
        for train_sample in getattr(protocol,phase)():
            file = dict()
            for key,value in train_sample.items():
                if key=="annotated":
                    value = [segment for segment in value if segment.duration>self.duration]
                    file['annotated_duration'] = sum([segment.duration for segment in value])
                else:
                    pass
                    
                file[key]=value
            self.data.append(file)

    def prepare_chunk(
        self,
        file,
        chunk
    ):
        sample = dict()
        audio,sr = librosa.load(file["audio"],sr=self.sampling_rate)
        start,end = chunk.start*sr,chunk.end*sr
        sample["X"] = np.array(audio[math.ceil(start):math.ceil(end)])
        if len(sample["X"].shape)==1:
            sample["X"] = sample["X"].reshape(1,-1)
        sample['y'] = file['annotation'].discretize(chunk,duration=self.duration)
        return sample

    def select_chunk(
        self,
        rng
        ):
        while True:
            file = rng.choices(self.data,
                            weights=[sample['annotated_duration'] for sample in self.data],
                            k=1)[0]
            segment = rng.choices(file['annotated'],
                                weights=[segment.duration for segment in file['annotated']],
                                k=1)[0]
            
            start_time = rng.uniform(segment.start,segment.end-self.duration)
            chunk = Segment(start_time,start_time+self.duration)
            yield self.prepare_chunk(file,chunk)

    def __iter__helper(
        self,
    ):
        rng = random_generation()   ##not reproducible
        while True:
  
            chunks = self.select_chunk(rng)
            yield next(chunks)

    def __iter__(self):
        return self.__iter__helper()

    def __len__(self):
        return sum([file["annotated_duration"] for file in self.data])//self.duration

    
class AMICollate:

    def __init__(
        self,
        ):
        pass 

    def __call__(
        self,
        batch
    ):
       
        output = {"X":[],"y":[]}
        for b in batch:
            output["X"].append(b['X'])
        
        labels = list(set(itertools.chain(*(b["y"].labels for b in batch))))
        y_batch = torch.zeros((len(batch),batch[0]["y"].data.shape[0],len(labels)))

        for b,sample in enumerate(batch):
            for local_idx,label in enumerate(sample["y"].labels):
                global_idx = labels.index(label)
                y_batch[b,:,global_idx] = torch.from_numpy(sample["y"].data[:,local_idx])

        output["X"] = torch.from_numpy(np.array(output["X"]))
        output["y"] = y_batch
          
        return output
    


