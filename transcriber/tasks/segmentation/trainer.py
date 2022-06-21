import torch
from pyannote.database import Protocol
from typing import Optional
from torch.utils.data import DataLoader

from transcriber.tasks.utils import min_value_check
from transcriber.tasks.segmentation.dataloader import AMIDataset,AMICollate
from transcriber.tasks.segmentation.model import SincNet


class Trainer:

    def __init__(
        self,
        protocol:Protocol,
        duration:float=2.0,
        batch_size:int=16,
        epochs:int=5,
        learning_rate:float=1e-5,
        device:str="cpu",
        sampling_rate:int=16000
    ):
        self.protocol = protocol
        self.duration = min_value_check(duration,1.0)
        self.batch_size = min_value_check(batch_size,2)
        self.epochs = min_value_check(epochs,1)
        self.learning_rate = learning_rate
        self.sampling_rate = sampling_rate

        if device not in ("cpu","cuda"):
            raise ValueError("device should be cpu or cuda")
        else:
            if getattr(torch,device).is_available():
                self._device = torch.device(device)
            else:
                raise ValueError(f"{device} not available!")

    def train(
        self,
        experiment_name:Optional[str]=None,
        run_name:Optional[str]=None
    ):
        pass
        self.experiment_name = experiment_name if not None else "experiment"
        self.run_name = run_name if not None else "run"


    def _run_single_batch(
        self
    ):
        pass

    def _prepare_dataloaders(
        self
    ):
        collate_fn = AMICollate()
        train_dataset = AMIDataset(protocol=self.protocol,duration=self.duration,
                            sampling_rate=self.sampling_rate,phase="train")
        train_dataset = DataLoader(train_dataset,batch_size=self.batch_size,collate_fn=collate_fn)

        dev_dataset = AMIDataset(protocol=self.protocol,duration=self.duration,
                            sampling_rate=self.sampling_rate,phase="development")
        dev_dataset = DataLoader(train_dataset,batch_size=self.batch_size,collate_fn=collate_fn)

        return {"train":train_dataset,
                "development":dev_dataset}