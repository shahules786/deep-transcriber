import itertools
import numpy as np
import librosa
import torch
import math
from torch.utils.data import IterableDataset
from pyannote.core import Segment

from transcriber.tasks.utils import softmax,random_generation
from transcriber.tasks.segmentation.model import MODEL_OUTPUT_FRAMES

class AMIDataset(IterableDataset):

    def __init__(
        self,
        protocol,
        duration=2,
        sampling_rate=16000,
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

        self.resolution_msec = self.duration/MODEL_OUTPUT_FRAMES

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
        sample['y'] = file['annotation'].discretize(chunk,duration=self.duration,resolution=self.resolution_msec)
        return sample

    def select_chunk(
        self,
        rng
    ):
        i=0
        while True:
            i+=1
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
        chunks = self.select_chunk(rng)
        while True:

            yield next(chunks)

    def __iter__(self):
        return self.__iter__helper()

    def __len__(self):
        return sum([file["annotated_duration"] for file in self.data])//self.duration

    
class AMICollate:

    def __init__(
        self,
        max_num_speakers:int
    ):
        self.max_num_speakers = max_num_speakers

    def prepare_target(
        self,
        target:torch.tensor
    ):
        num_speakers = target.shape[-1]
        max_num_speakers_framelevel = torch.sum(target.sum(1)>0,dim=1)
        max_speakers_batch = torch.max(max_num_speakers_framelevel)
        speaker_activity_indices = torch.argsort(target.sum(dim=1),dim=1,descending=True)

        new_target = torch.zeros(target.shape[0],target.shape[1],
                                max(self.max_num_speakers,max_speakers_batch), 
                                    dtype=target.dtype, device=target.device)

        for b,indices in enumerate(speaker_activity_indices):
            for i,index in zip(range(max_speakers_batch),indices):
                new_target[b,:,i] = target[b,:,index]

        return new_target

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
        output["y"] = self.prepare_target(y_batch)
          
        return output
    


