import torch
from pyannote.database import Protocol
from typing import Optional
from torch.utils.data import DataLoader
from torch.optim import Adam
import mlflow 
import logging
import numpy as np


from transcriber.tasks.utils import min_value_check
from transcriber.tasks.segmentation.loss import PermutationInvarientTraining, losses
from transcriber.tasks.segmentation.dataloader import AMIDataset,AMICollate
from transcriber.tasks.segmentation.model import SegmentNet


class Trainer:

    def __init__(
        self,
        protocol:Protocol,
        duration:float=5.0,
        batch_size:int=16,
        epochs:int=1,
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
                self.device = torch.device(device)
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

        model = SegmentNet().to(self.device)
        optimizer = Adam(lr=self.learning_rate,params=model.parameters())
        bce_loss = losses(loss="bce")
        Perumtation_bce = PermutationInvarientTraining(loss="bce")
        dataloaders = self._prepare_dataloaders()
        
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if not experiment:
            experiment = mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=self.run_name):

            for epoch in range(self.epochs):
                loss_dict = {"train":[],"developement":[]}

                phase = "train"
                for batch,data in enumerate(dataloaders["train"]):
                    predition,batch_loss_dict = self._run_single_batch(
                        model=model,data=data,optimizer=optimizer,
                        loss_obj=bce_loss,Permutation=Perumtation_bce,phase=phase
                    )
                    loss_dict[phase].append(batch_loss_dict["total_loss"])
                
                phase = "developement"
                for batch,data in enumerate(dataloaders["train"]):
                    predition,batch_loss_dict = self._run_single_batch(
                        model=model,data=data,optimizer=optimizer,
                        loss_obj=bce_loss,Permutation=Perumtation_bce,phase=phase
                    )
                    loss_dict[phase].append(batch_loss_dict["total_loss"])

                logging.info(f"Train loss epoch {epoch} : {np.mean(loss_dict['train'])}")
                logging.info(f"Valid loss epoch {epoch} : {np.mean(loss_dict['developement'])}")
                
                mlflow.log_metrics({"Train Loss":np.mean(loss_dict['train'])},step=epoch)
                mlflow.log_metrics({"Valid Loss":np.mean(loss_dict['developement'])},step=epoch)

                    
    def _run_single_batch(
        self,
        data,
        model,
        optimizer,
        Permutation,
        loss_obj,
        phase="train"
    ):

        if phase == "train":
            model.train()
            optimizer.zero_grad()
        else:
            model.eval()

        input_waveform, target = data.items()
        prediction = model(input_waveform)
        min_weight_perumuation, _ = Permutation(prediction,target.data)
        seg_loss = loss_obj.segmentation_loss(min_weight_perumuation,target.data)
        vad_loss = loss_obj.vad_loss(min_weight_perumuation,target.data)
        loss = seg_loss + vad_loss

        if phase == "train":
                loss.backward()
                optimizer.step()
        else:
            pass


        return {"prediction":prediction,
                "loss":{"seg_loss":seg_loss.item(),"vad_loss":vad_loss.item(),
                        "total_loss":loss.item()}
                }

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