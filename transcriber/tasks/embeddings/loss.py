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
        self.w = nn.parameter.Parameter(torch.tensor(10.0))
        self.b = nn.parameter.Parameter(torch.tensor(-5.0))

    def forward(
        self,
        embeddings,
    ):
        centroids = embeddings.mean(dim=1)
        sum_ = embeddings.sum(dim=1)
        e = embeddings.view(self.N*self.M,-1)
        cosine_sim = torch.mm(e,centroids.transpose(0,1))
        
        for j in range(self.N):
            for i in range(self.M):
                cj = (sum_[j] - e[j*self.M + i]) / (self.M - 1)
                cosine_sim[j*self.M + i][j] = torch.dot(cj, e[j*self.M + i])
        cosine_sim = self.w*cosine_sim + self.b 
        labels = torch.zeros((self.N*self.M,),dtype=int,device=embeddings.device)
        for i,j in enumerate(range(0,self.N*self.M,self.M)):
            labels[j:j+self.M] = i
        loss = cross_entropy(cosine_sim,labels)
        return loss
        
    
    def equal_error_rate(embeddings_1,embeddings_2,N,M):
        
        centroid = embeddings_2.mean(dim=1)
        sum_ = embeddings_2.sum(dim=1)
        e = embeddings_1.view(N*M,-1)
        cosine_sim = torch.mm(e,centroid.transpose(0,1))
        for j in range(N):
                for i in range(M):
                    cj = (sum_[j] - e[j*M + i]) / (M - 1)
                    cosine_sim[j*M + i][j] = torch.dot(cj, e[j*M + i])

        labels = torch.zeros((N*M,),dtype=int,device=embeddings_1.device)
        for i,j in enumerate(range(0,N*M,M)):
            labels[j:j+M] = i
        
        eer_array = torch.stack([cosine_sim.max(dim=-1),labels,cosine_sim.argmax(dim=-1)])



        
