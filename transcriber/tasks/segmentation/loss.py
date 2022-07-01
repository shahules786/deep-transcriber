from torch import nn
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

PREDEFINED_LOSS = {"mse":nn.MSELoss(reduction="none"),
                    "bce":nn.BCELoss(reduction="none"),
                    }

class PermutationInvarientTraining:

    def __init__(
        self,
        loss:str="bce"
    ):
        if PREDEFINED_LOSS.get(loss) is not None:
            self.loss = PREDEFINED_LOSS[loss]
        else:
            raise ValueError("Invalid Loss")

    def __call__(
        self,
        y1:torch.tensor,
        y2:torch.tensor
    ):
        batch_size_y1, num_frames_y1, num_classes_y1 = y1.shape
        batch_size_y2, num_frames_y2, num_classes_y2 = y2.shape
        if batch_size_y1!=batch_size_y2:
            raise ValueError(f"batch size y1 {batch_size_y1} must be equal to batch size y2 {batch_size_y2}")
        
        if num_frames_y1!=num_frames_y2:
            raise ValueError(f"number of frames y1 {num_frames_y1} must be equal to number of frames y2 {num_frames_y2}")
        
        permutated_y2 = torch.zeros(y1.shape, device=y2.device, dtype=y2.dtype,)
        permutations = []
        for b,(y1_,y2_) in enumerate(zip(y1,y2)):
            with torch.no_grad():
                cost_matrix = torch.stack(
                    [torch.mean(self.loss(y2_,y1_[:,i:i+1].expand(-1,num_classes_y2)),dim=0) 
                        for i in range(num_classes_y1)]
                )

            if num_classes_y1<num_classes_y2:
                pad = (0,0,0,num_classes_y2-num_classes_y1)
                cost_matrix = F.pad(
                        input=cost_matrix,
                        pad=pad,
                        value=torch.max(cost_matrix)+1
                )

            permutation = [None] * num_classes_y1
            for k1,k2 in zip(*linear_sum_assignment(cost_matrix.cpu())):
                if k1<num_classes_y1:
                    permutated_y2[b,:,k1] = y2_[:,k2]
                    permutation[k1] = k2
                permutations.append(tuple(permutation))

        return permutated_y2, permutations


            
class losses:

    def __init__(
        self,
        loss:str="bce"
    ):

        self.loss = loss

    def segmentation_loss(
        self,
        input:torch.tensor,
        target:torch.tensor
    ):

        if self.loss == "bce":
            return F.binary_cross_entropy(input,target.float())
        elif self.loss == "mse":
            return F.mse_loss(input, target.float())
        else:
            raise ValueError("loss should be either 'bce' or 'mse'")

    def vad_loss(
        self,
        input:torch.tensor,
        target:torch.tensor
    ):
        
        input,_ = torch.max(input,dim=2,keepdim=True)
        target,_ = torch.max(target,dim=2,keepdim=True)

        if self.loss == "bce":
            return F.binary_cross_entropy(input, target)
        elif self.loss == "mse":
            return F.mse_loss(input, target)
        else:
            raise ValueError("loss should be either 'bce' or 'mse'")






        