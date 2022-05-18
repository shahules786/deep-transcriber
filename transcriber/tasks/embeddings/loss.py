import torch
from torch import nn
from torch.nn.functional import cross_entropy


class Ge2eLoss(nn.Module):

    def __init__(
        self,

    ):
        self.w = nn.parameter(torch.tensor(5.0))
        self.b = nn.parameter(torch.tensor(1.0))

    def forward(
        self,
        embeddings,
        N:int,
        M:int
    ):
        centroids = embeddings.mean(dim=1)
        sum_ = embeddings.sum(dim=1)
        cosine_sim = torch.mm(embeddings.reshape(N*M,-1),centroids.transpose(0,1))
        for j in range(N):
            for k in range(M):
                c_j = sum_[j] - embeddings[j,k,:] / (1-M)
                cosine_sim[M*j+k][j] = torch.dot(c_j,embeddings[j,k,:])
        
        cosine_sim = self.w*cosine_sim + self.b 
        labels = torch.zeros((N*M,),dtype=int)
        for i,j in enumerate(range(0,N*M,M)):
            labels[j:j+M] = i

        loss = cross_entropy(cosine_sim,labels)
        return loss
        
        
