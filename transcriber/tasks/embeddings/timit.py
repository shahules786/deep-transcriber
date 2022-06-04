import numpy as np
from typing import Optional
import os
import logging
import glob 
import librosa 
from argparse import ArgumentParser
from tqdm import tqdm
import shutil
logging.getLogger().setLevel(logging.INFO)



class ProcessTimit:

    def __init__(
        self,
        directory:str,
        num_dilects:int,
        output:str,
        n_fft:int,
        win_length:int,
        hop_length:int,
        sampling_rate:int,
        frames:int
        ):

        self.directory = directory
        self.num_dilects = num_dilects
        self.output = output
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate
        self.frames = frames
        self.min_utterance_len = self.frames*self.hop_length + self.win_length

        if not os.path.exists(self.output):
            os.mkdir(self.output)
            logging.info("Directory created..")
        else:
            logging.info("DELETING OLD FILES..")
            for file in glob.glob(os.path.join(self.output,"*")):
                os.remove(file)


    def process_data(
        self,
        ):
        n=0
        for i,dilect_dir in enumerate(os.listdir(self.directory)):
            if os.path.isdir(os.path.join(self.directory,dilect_dir)) and n<=self.num_dilects:
                logging.info(f"DILECT {i} is {dilect_dir}")
                n+=1
                for j,speaker_dir in enumerate(os.listdir(os.path.join(self.directory,dilect_dir))):
                        if os.path.isdir(os.path.join(self.directory,dilect_dir,speaker_dir)):
                            utterances = self.process_wav(os.path.join(self.directory,dilect_dir,speaker_dir))
                            logging.info(f"{utterances.shape[0]} utterances found..")
                            np.save(os.path.join(self.output,f"{dilect_dir}_{speaker_dir}.npy"),utterances)


    def process_wav(
        self,
        files:str
    ):
        utterances = []
        for file in [file for file in glob.glob(os.path.join(files,"*")) if file.endswith(".WAV")]:
            audio,sr = librosa.core.load(file,sr=self.sampling_rate)
            audio_intervals = librosa.effects.split(audio,top_db=30)
            for interval in audio_intervals:
                if interval[1]-interval[0] > self.min_utterance_len:
                    audio_utter = audio[interval[0]:interval[1]]
                    mel_spec = np.log10(librosa.feature.melspectrogram(y=audio_utter,
                    sr=sr,n_fft=self.n_fft,hop_length=self.hop_length,win_length=self.win_length,n_mels=40))
                    utterances.append(mel_spec[:,:self.frames])
                    utterances.append(mel_spec[:,-self.frames:])
        return np.array(utterances)
                
if __name__ == "__main__":

    parser = ArgumentParser(description="argument parser for TIMIT")
    parser.add_argument("--directory", type=str, help="Directory with files")  #/Users/shahules/Myprojects/deep-transcriber/data/lisa/data/timit/raw/TIMIT
    parser.add_argument("--output", type=str, help="Directory to write output files")    #/Users/shahules/Myprojects/deep-transcriber/data/output
    parser.add_argument("--sampling_rate", type=int, help="Sampling rate", default=16000)
    parser.add_argument("--num_dilects", type=int, help="Number of dialects to process", default=8)
    parser.add_argument("--n_fft", type=int, help="N fft", default=512)
    parser.add_argument("--hop_length", type=int, help="Hop length for STFT", default=160)
    parser.add_argument("--win_length", type=int, help="Window length for STFT", default=400)
    parser.add_argument("--frames", type=int, help="mel spectrogram frames to save", default=180)
    args = parser.parse_args().__dict__
    timit_object = ProcessTimit(**args)
    timit_object.process_data()
    logging.info("PROCESS COMPLETED!")









                





            

        






