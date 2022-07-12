from typing import Optional
import torch
import torch.nn.functional  as F

from transcriber.utils.io import Audio
from transcriber.tasks.segmentation.model import SegmentNet

class Segmenter:

    def __init__(
        self,
        model:str,
        duration:int=5,
        step:Optional[float]=None,
        sampling_rate:int=16000,
        aggregate:bool=True,
        batch_size:int=32
    ):

        self.model = SegmentNet(max_num_speakers=4)
        self.model.eval()
        self.model.load_state_dict(torch.load(model,map_location=torch.device('cpu')))
        self.audio = Audio(sampling_rate=sampling_rate,mono=True,return_tensor=True)
        self.sampling_rate = sampling_rate
        self.duration = duration 
        self.step = step if ((step!=None) and (step<self.duration)) else 0.1 * self.duration
        self.batch_size = batch_size
    
    def __call__(
        self,
        audio,
        sampling_rate

    ):
        """
        audio : can be path to wav file or audio data
        """
        audio = self.audio(audio,sampling_rate)
        audio_chunks = self.prepare_chunks(audio)
        return self.batch_infer(audio_chunks)
        

    def batch_infer(
        self,
        batch
    ):
        model_outputs = []
        for batch_no in range(0,batch.shape[0],self.batch_size):
            chunk = batch[batch_no:batch_no+self.batch_size]
            output = self.model(chunk)
            model_outputs.append(output)
        return model_outputs

    def prepare_chunks(
        self,
        audio
    ):

        window_size = round(self.duration*self.sampling_rate)
        step_size = round(self.step*window_size)
        num_samples = len(audio)

        audio = audio.squeeze()
        chunks = audio.unfold(dimension=0,size=window_size,step=step_size)
        num_chunks,chunk_size = chunks.shape

        if (num_samples - window_size)%step_size !=0:
            final_chunk = audio[num_chunks*step_size:]
            final_chunk_padded = F.pad(final_chunk,(0,chunk_size-final_chunk.shape[0]),
                                        value=0.0).unsqueeze(0)
            chunks = torch.cat([chunks,final_chunk_padded])

        return chunks
                                        




