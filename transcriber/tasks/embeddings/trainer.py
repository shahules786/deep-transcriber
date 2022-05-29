
import logging
import os
import yaml 
import torch
import numpy as np
import mlflow
from torch.utils.data import DataLoader
from torch.optim import Adam,SGD

from transcriber.tasks.embeddings.dataloader import TimitDataset,TimitCollate
from transcriber.tasks.embeddings.model import Embeder
from transcriber.tasks.utils import min_value_check, path_check
from transcriber.tasks.embeddings.loss import Ge2eLoss


class EmbedTrainer:

    def __init__(
        self,
        input_size:int,
        hidden_size:int,
        num_layers:int,
        embedding_dim:int,
        model_dir : str,
        logger:str = "DEBUG"
    ):
        if min_value_check(input_size,0):
            self.input_size = input_size

        if min_value_check(num_layers,0):
            self.num_layers = num_layers

        if min_value_check(hidden_size,0):
            self.hidden_size = hidden_size

        if min_value_check(embedding_dim,0):
            self.embedding_dim = embedding_dim

        self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if not os.path.exists(model_dir):
            logging.info(f"Creating {model_dir}...")
            os.mkdir(model_dir)

        if path_check(model_dir):
            self.model_dir = model_dir

        if logger in ("DEBUG","INFO","WARN"):
            logging.basicConfig(filename='deep-transcriber.log', filemode='w',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S', level=logging.DEBUG,)
            logging.basicConfig(level=getattr(logging,logger))
            
           
    @property
    def device(self):
        return self._device

    @device.setter
    def device(self,device):

        if device not in ("cpu","cuda"):
            raise ValueError("device should be cpu or cuda")
        else:
            if getattr(torch,device).is_available():
                self._device = torch.device(device)
            else:
                raise ValueError(f"{device} not available!")
    
    def train(
        self,
        train:str,
        test:str,
        batch_size:int,
        epochs:int,
        lr:float,
        n_speakers:int,
        n_utterances:int,
        experiment_name:str,
        run_name:str

        
    ):
        if path_check(train):
            self.train = train

        if path_check(test):
            self.test = test

        if min_value_check(batch_size,0):
            self.batch_size = batch_size

        if min_value_check(epochs,0):
            self.epochs = epochs

        if min_value_check(n_speakers,0):
            self.n_speakers = n_speakers

        if min_value_check(n_utterances,0):
            self.n_utterances = n_utterances
        
        self.lr = lr

        self.experiment_name = experiment_name if not None else "experiment"
        self.run_name = run_name if not None else "run"

        datalaoders = self._prepare_dataloaders()
        model = Embeder(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers,
                        embed_size=self.embedding_dim).to(self.device)
        loss_fn = Ge2eLoss(N=self.n_speakers,M=self.n_utterances)
        optimizer = SGD(self._get_optimizer(model,loss_fn))
        

        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if not experiment:
            experiment = mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=self.run_name):
            
            for epoch in range(self.epochs):
                loss = {"train":[], "valid": []}
                for batch_num,data in enumerate(datalaoders['train']):
                    output = self._run_single_batch(model,optimizer,loss_fn,data,phase="train")
                    loss['train'].append(output['loss'])
                
                print("valid phase")
                for batch_num,data in enumerate(datalaoders['valid']):
                    output = self._run_single_batch(model,optimizer,loss_fn,data=data,phase="valid")
                    loss['valid'].append(output['loss'])
                
                logging.info(f"Train loss epoch {epoch} : {np.mean(loss['train'])}")
                logging.info(f"Valid loss epoch {epoch} : {np.mean(loss['valid'])}")
                
                mlflow.log_metrics({"Train Loss":np.mean(loss['train'])},step=epoch)
                mlflow.log_metrics({"Valid Loss":np.mean(loss['valid'])},step=epoch)


            logging.info("Training Finished. Saving model..")
            torch.save(model.state_dict(),os.path.join(self.model_dir,"model.pt"))
            mlflow.log_artifact(os.path.join(self.model_dir,"model.pt"))
            if os.path.exists("deep-transcriber.log"):
                mlflow.log_artifact('deep-transcriber.log')
                
    def _run_single_batch(
        self,model,optimizer,criterion,data,phase
    ):
        data["data"] = data["data"].to(self.device)

        if phase == "train":
            model.train()
            optimizer.zero_grad()
        else:
            model.eval()

        with torch.set_grad_enabled(phase == "train"):
            embeddings = model(data["data"])
            embeddings = embeddings[data["unpermute"]].reshape(self.n_speakers,self.n_utterances,self.embedding_dim)
            loss = criterion(embeddings)
            if phase == "train":
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                torch.nn.utils.clip_grad_norm_(criterion.parameters(), 1.0)
                optimizer.step()
            
        return {"embeddings":embeddings,"loss":loss.item()}
            

    def _get_optimizer(
        self,
        model,
        loss
    ):
        optimizer_params = [
            {"params":model.parameters(),
            "weight_decay":0.0,
            "lr":self.lr},

            {"params":loss.parameters(),
            "weight_decay":0.0,
            "lr":self.lr},

        ]
        print(optimizer_params)
        return optimizer_params

    def _prepare_dataloaders(
        self,
    ):
        if self.batch_size!=self.n_speakers:
            raise ValueError("batch_size should be equal to n_speakers")

        train_dataset = TimitDataset(directory=self.train, n_utterances=self.n_utterances,
                                        n_speakers = self.n_speakers)
        collate_fn = TimitCollate(n_speakers = self.n_speakers, n_utterances=self.n_utterances)
        train_dataset = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)

        valid_dataset = TimitDataset(directory=self.train, n_utterances=self.n_utterances,
                                        n_speakers = self.n_speakers)
        valid_dataset = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)

        return {"train":train_dataset,
                "valid":valid_dataset}


if __name__ == "__main__":

    with open('transcriber/tasks/embeddings/conf.yaml') as file:
        args = yaml.full_load(file)

    trainer = EmbedTrainer(input_size=args["model"]["input_size"],
                            hidden_size=args["model"]["hidden_size"],
                            num_layers=args["model"]["num_layers"],
                            embedding_dim=args["model"]["embedding_dim"],
                            model_dir=args["model"]["model_dir"],
                            logger=args["data"]["logger"]) 
    
    trainer.train(train=args["data"]["train"],
                test=args["data"]["test"],
                batch_size=args["training"]["batch_size"],
                epochs=args["training"]["epochs"],
                lr=args["training"]["lr"],
                n_speakers=args["training"]["n_speakers"],
                n_utterances=args["training"]["n_utterances"],
                experiment_name=args["training"]["experiment_name"],
                run_name=args["training"]["run_name"])





