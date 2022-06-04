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
        
    with torch.no_grad():
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
        
        eers=0.0
        for i in range(N):
            y, y_score = (labels==i).int(),cosine_sim[:,i]
            eer,thresh = calculate_eer(y,y_score)
            eers += eer
        return eers/N        
    
def calculate_eer(y, y_score, pos=1):
    """ 
    Method to compute eer, retrieved from https://github.com/a-nagrani/VoxSRC2020/blob/master/compute_EER.py 
    `y` is tensor of (cnt, ) of labels (0 or 1)
    `y_score` is tensor of (cnt, ) of similarity scores
    `pos` is the positive label, 99% of the time leave it as 1.
    """
    try:
        from scipy.interpolate import interp1d
        from scipy.optimize import brentq
        from sklearn.metrics import roc_curve
    except ModuleNotFoundError: 
        raise ModuleNotFoundError("Problem: for EER metrics, you require scipy and sklearn. Please install them first.")
    y = y.cpu().numpy()
    y_score = y_score.cpu().numpy()
    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=pos)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh

        
