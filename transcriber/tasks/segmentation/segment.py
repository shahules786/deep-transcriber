from typing import Optional
import torch
import torch.nn.functional  as F

from transcriber.utils.io import Audio

class Segmenter:

    def __init__(
        self,
        duration:int=5,
        step:Optional[float]=None,
        sampling_rate:int=16000,
        aggregate:bool=True
    ):
        self.audio = Audio(sampling_rate=sampling_rate,mono=True,return_tensor=True)
        self.sampling_rate = sampling_rate
        self.duration = duration 
        self.step = step if ((step!=None) and (step<self.duration)) else 0.1 * self.duration
    
    def __call__(
        self,
        audio

    ):
        """
        audio : can be path to wav file or audio data
        """
        audio = self.audio(audio)
        audio_chunks = self.prepare_chunks(audio)


    def prepare_chunks(
        self,
        audio
    ):

        window_size = round(self.duration*self.sampling_rate)
        step_size = round(self.step*window_size)
        num_samples = len(self.audio)

        audio = audio.squeeze()
        chunks = audio.unfold(dimension=0,size=window_size,step=step_size)
        num_chunks,chunk_size = chunks.shape

        if (num_samples - window_size)%step_size !=0:
            final_chunk = audio[num_chunks*step_size:]
            final_chunk_padded = F.pad(final_chunk,(0,chunk_size-final_chunk.shape[0]),
                                        value=0.0).unsqueeze(0)
            chunks = torch.cat([chunks,final_chunk_padded])

        return chunks
                                        




