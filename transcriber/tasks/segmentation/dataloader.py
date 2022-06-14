from random import sample
from torch.utils.data import IterableDataset
import numpy as np
import librosa
from pyannote.core import Segment

from transcriber.tasks.utils import softmax




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

    def get_chunks(
        self,
        file,
        segment
    ):
        sample = dict()
        start_time = np.random.uniform(segment.start,segment.end-self.duration)
        chunk = Segment(start_time,start_time+self.duration)
        audio,sr = librosa.load(file["audio"],sr=self.sampling_rate)
        start,end = chunk.start*sr,chunk.end*sr
        sample["X"] = audio[int(start):int(end)]
        sample['y'] = file['annotation'].discretize(chunk,duration=self.duration)
        print(sample["X"].shape,sample['y'].data.shape)
        yield sample

    def __iter__helper(
        self,
    ):

        while True:

            file = np.random.choice(self.data,p=softmax([sample['annotated_duration'] for sample in self.data]))
            segment = np.random.choice(file['annotated'],
                                p=softmax([segment.duration for segment in file['annotated']])
                                )
        
            chunks = self.get_chunks(file,segment)
            yield next(chunks)

    def __iter__(self):
        return self.__iter__helper()

    
    