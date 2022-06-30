import torch
from pyannote.database import Protocol
from typing import Optional
from torch.utils.data import DataLoader
from torch.optim import Adam
import mlflow 
import logging
import numpy as np
import yaml
import os


from transcriber.tasks.utils import min_value_check
from transcriber.tasks.segmentation.loss import PermutationInvarientTraining, losses
from transcriber.tasks.segmentation.dataloader import AMIDataset,AMICollate
from transcriber.tasks.segmentation.model import SegmentNet

logging.getLogger().setLevel(logging.INFO)


class Trainer:

    def __init__(
        self,
        protocol:Protocol,
        duration:float=5.0,
        max_num_speakers=4,
        batch_size:int=16,
        epochs:int=1,
        learning_rate:float=1e-5,
        device:str="cpu",
        sampling_rate:int=16000
    ):
        self.protocol = protocol
        self.max_num_speakers = max_num_speakers
        self.duration = min_value_check(duration,1.0,"duration")
        self.batch_size = min_value_check(batch_size,2,"batch size")
        self.epochs = min_value_check(epochs,1,"epochs")
        self.learning_rate = learning_rate
        self.sampling_rate = sampling_rate

        if device not in ("cpu","cuda"):
            raise ValueError("device should be cpu or cuda")
        elif device == "cuda":
            if getattr(torch,device).is_available():
                self.device = torch.device(device)
            else:
                raise ValueError(f"{device} not available!")
        else:
            self.device = torch.device(device)

    def train(
        self,
        experiment_name:Optional[str]=None,
        run_name:Optional[str]=None
    ):
        pass
        self.experiment_name = experiment_name if not None else "experiment"
        self.run_name = run_name if not None else "run"

        model = SegmentNet(self.max_num_speakers).to(self.device)
        optimizer = Adam(lr=self.learning_rate,params=model.parameters())
        bce_loss = losses(loss="bce")
        Perumtation_bce = PermutationInvarientTraining(loss="bce")
        dataloaders = self._prepare_dataloaders()
        
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if not experiment:
            experiment = mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=self.run_name):
            num_train_batches = int(dataloaders["train"].dataset.__len__()//self.batch_size)
            num_dev_batches = int(dataloaders["development"].dataset.__len__()//self.batch_size)
            logging.info(f"Number of train steps per epoch = {num_dev_batches}")
            logging.info(f"Number of validation steps per epoch = {num_dev_batches}")

            for epoch in range(self.epochs):
                loss_dict = {"train":[],"development":[]}

                phase = "train"
                for batch_num,data in zip(range(num_train_batches), dataloaders[phase]):
                    if data["y"].shape[-1] <= self.max_num_speakers:
                        predition,batch_loss_dict = self._run_single_batch(
                            model=model,data=data,optimizer=optimizer,
                            loss_obj=bce_loss,Permutation=Perumtation_bce,phase=phase
                        )
                        loss_dict[phase].append(batch_loss_dict["loss"]["total_loss"])
                        logging.info(f'Train loss {batch_num} : {batch_loss_dict["loss"]["total_loss"]}')
                        mlflow.log_metrics({"Train Loss":batch_loss_dict["loss"]["total_loss"]},step=batch_num)

                        
                
                phase = "development"
                for batch_num,data in zip(range(num_dev_batches),dataloaders[phase]):
                    predition,batch_loss_dict = self._run_single_batch(
                        model=model,data=data,optimizer=optimizer,
                        loss_obj=bce_loss,Permutation=Perumtation_bce,phase=phase
                    )
                    loss_dict[phase].append(batch_loss_dict["loss"]["total_loss"])
                    logging.info(f'Valid loss {batch_num} : {batch_loss_dict["loss"]["total_loss"]}')
                    mlflow.log_metrics({"Valid Loss":batch_loss_dict["loss"]["total_loss"]},step=batch_num)


                logging.info(f"Train loss epoch {epoch} : {np.mean(loss_dict['train'])}")
                logging.info(f"Valid loss epoch {epoch} : {np.mean(loss_dict['development'])}")
                
                mlflow.log_metrics({"Train Loss Epochs":np.mean(loss_dict['train'])},step=epoch)
                mlflow.log_metrics({"Valid Loss Epochs":np.mean(loss_dict['development'])},step=epoch)

                    
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

        with torch.set_grad_enabled(phase == "train"):
            input_waveform, target = data.values()
            prediction = model(input_waveform)
            min_weight_perumuation, _ = Permutation(target.data,prediction)
            seg_loss = loss_obj.segmentation_loss(min_weight_perumuation,target.data)
            vad_loss = loss_obj.vad_loss(min_weight_perumuation,target.data)
            loss = seg_loss + vad_loss

            if phase == "train":
                loss.backward()
                optimizer.step()

        return (prediction.cpu(),
                {
                "loss":{"seg_loss":seg_loss.item(),"vad_loss":vad_loss.item(),
                        "total_loss":loss.item()}
                })

    def _prepare_dataloaders(
        self
    ):
        collate_fn = AMICollate(max_num_speakers=self.max_num_speakers)
        train_dataset = AMIDataset(protocol=self.protocol,duration=self.duration,
                            sampling_rate=self.sampling_rate,phase="train")
        train_dataset = DataLoader(train_dataset,batch_size=self.batch_size,collate_fn=collate_fn)

        dev_dataset = AMIDataset(protocol=self.protocol,duration=self.duration,
                            sampling_rate=self.sampling_rate,phase="development")
        dev_dataset = DataLoader(dev_dataset,batch_size=self.batch_size,collate_fn=collate_fn)

        return {"train":train_dataset,
                "development":dev_dataset}


if __name__ == "__main__":

    with open('transcriber/tasks/segmentation/conf.yaml') as file:
        args = yaml.full_load(file)

    from pyannote.database import add_custom_protocols
    from pyannote.database import FileFinder
    
    os.environ["PYANNOTE_DATABASE_CONFIG"] = args["Data"]["database"]

    preprocessors = {'audio': FileFinder()}
    database,task = add_custom_protocols()
    name = 'AMI.SpeakerDiarization.only_words'
    database_name, task_name, protocol_name = name.split(".")
    protocol = database[database_name]().get_protocol(
        task=task_name,protocol=protocol_name, preprocessors=preprocessors
    )
    protocol.name = name

    trainer = Trainer(
        protocol=protocol,
        duration=args["Training"]["duration"],
        max_num_speakers=args["Training"]["max_num_speakers"],
        batch_size=args["Training"]["batch_size"],
        epochs=args["Training"]['epochs'],
        learning_rate=args["Training"]["learning_rate"],
        sampling_rate=args["Training"]["sampling_rate"],
        device=args["Training"]["device"]
    )
    trainer.train(experiment_name=args["Training"]["experiment_name"],
                run_name=args["Training"]["run_name"])
    