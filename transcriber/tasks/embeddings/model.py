import torch.nn as nn
import torch

class Embeder(nn.Module):

    def __init__(
        self,
        embed_size:int,
        input_size:int,
        hidden_size:int,
        num_layers:int = 1,
        batch_first:bool = True
    ):
        super(self,Embeder).__init__()
        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size,num_layers=num_layers,
                            batch_first=batch_first)
        self.projection = nn.Linear(hidden_size,embed_size)

        def forward(
            self,
            input
        ):
            ouput,(h_c,c_n) = self.LSTM(input)
            embedding = self.projection(ouput[:,-1])
            embedding = embedding / torch.norm(embedding,dim=1) 
            return embedding

