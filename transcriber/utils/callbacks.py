import torch
import logging
import os

class EarlyStopping:

    def __init__(
        self,
        patience:int=10,
        mode="min",
        filename="model.pth",
        directory="./model"

    ):
        self.patience = patience 
        self.counter = 0
        if mode in ("min","max"):
            self.mode = mode
        else:
            raise ValueError("mode must be either 'min' or 'max'")
        self.best_loss = None
        self.early_stop = False
        if not os.path.exists(directory):
            logging.info("CREATING DIR TO SAVE MODEL...")
            os.mkdir(directory)

        self.filepath = os.path.join(directory,filename)

    def __call__(
        self,
        loss,
        model

    ):
        if self.best_loss == None:
            self.best_loss = loss
            self.save_checkpoint(model)
        else:
            if self.mode == "min":
                if loss < self.best_loss:
                    self.best_loss = loss
                    self.save_checkpoint(model)
                else:
                    self.counter += 1
            else:
                if loss > self.best_loss:
                    self.best_loss = loss
                    self.save_checkpoint(model)
                else:
                    self.counter += 1


        if self.counter >= self.patience:
            print(f"Early stopping with best loss {self.best_loss}")
            self.early_stop = True

    def save_checkpoint(
        self,
        model
    ):
        logging.info(f"SAVING MODEL...{self.best_loss : .2f}")
        torch.save(model.state_dict(),self.filepath)


