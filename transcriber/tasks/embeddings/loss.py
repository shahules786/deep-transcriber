import torch
from torch import nn
from torch.nn.functional import cross_entropy


class Ge2eLoss(nn.Module):

    def __init__(
        self,
        N:int,
        M:int
    ):  
        super(Ge2eLoss,self).__init__()
        self.N = N
        self.M = M  
        self.w = nn.parameter.Parameter(torch.tensor(5.0))
        self.b = nn.parameter.Parameter(torch.tensor(1.0))

    def forward(
        self,
        embeddings,
    ):
        centroids = embeddings.mean(dim=1)
        sum_ = embeddings.sum(dim=1)
        cosine_sim = torch.mm(embeddings.reshape(self.N*self.M,-1),centroids.transpose(0,1))
        for j in range(self.N):
            for k in range(self.M):
                c_j = sum_[j] - embeddings[j,k,:] / (1-self.M)
                cosine_sim[self.M*j+k][j] = torch.dot(c_j,embeddings[j,k,:])
        
        cosine_sim = self.w*cosine_sim + self.b 
        labels = torch.zeros((self.N*self.M,),dtype=int,device=embeddings.device)
        for i,j in enumerate(range(0,self.N*self.M,self.M)):
            labels[j:j+self.M] = i

        loss = cross_entropy(cosine_sim,labels)
        return loss
        
        
