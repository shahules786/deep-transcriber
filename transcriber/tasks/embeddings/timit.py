import numpy as np
from typing import Optional
import os
import logging
import glob 
import librosa 


class ProcessTimit:

    def __init__(
        self,
        directory:str,
        num_dilects:int,
        output:str,
        win_length:int,
        hop_length:int,
        sampling_rate:int,
        frames:int
        ):

        self.directory = directory
        self.num_dilects = num_dilects
        self.output = output
        self.win_length = win_length
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate
        self.frames = frames

    def process_data(
        self,
        ):

        for i,dilect_dir in enumerate(os.listdir(self.directory)):
            logging.info(f"DILECT {i} is {dilect_dir}")
            for j,speaker_dir in enumerate(os.listdir(os.path.join(self.directory,dilect_dir))):
                    utterances = self.process_wav(os.path.join(self.directory,dilect_dir,speaker_dir))
                    logging.info(f"{utterances.shape[0]} utterances found..")
                    np.save(os.path.join(self.output,dilect_dir,f"{speaker_dir}.npy"),utterances)

    def process_wav(
        self,
        files:str
    ):
        utterances = []
        for file in [file for file in glob.glob(files) if file.endswith(".wav")]:
            audio,sr = librosa.core.load(file,sr=self.sampling_rate)
            audio_intervals = librosa.effects.split(audio,top_db=30)
            for interval in audio_intervals:
                if interval[1]-interval[0] > self.min_utterance_len:
                    audio_utter = audio[interval[0]:interval[1]]
                    mel_spec = librosa.feature.melspectrogram(audio_utter,
                    sr=sr,n_fft=self.n_fft,hop_length=self.hop_length,win_length=self.win_length)
                    utterances.append([mel_spec[:,:self.frame]])
                    utterances.append([mel_spec[:,-self.frame:]])
        return np.array(utterances)
                
                





            

        






