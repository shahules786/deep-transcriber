from unicodedata import bidirectional
from torch import dropout, nn
from torch.nn.functional import conv1d
import torch
import logging
import numpy as np

from transcriber.tasks.utils import hertz_to_mel, mel_to_hertz

class SincConv(nn.Module):

    def __init__(
        self,
        out_channels:int,
        kernel_size:int,
        inp_channels:int=1,
        stride:int=1,
        padding:int=0,
        dilation:int=1,
        freq_low:int=30,
        min_bandwidth_freq:int=50,
        sampling_rate:int=16000
    ):

        super(SincConv,self).__init__()

        if inp_channels!=1:
            raise ValueError("sincnet only supports mono channel")
        self.inp_channels = inp_channels
        self.out_channels = out_channels

        if  kernel_size%2==0:
            logging.warn("Kernel size must be odd")
            kernel_size+1
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.sampling_rate = sampling_rate
        self.freq_low = freq_low
        self.min_bandwidth_freq = min_bandwidth_freq

        low_freq_hertz = 30
        max_freq_hertz = self.sampling_rate/2 - (self.freq_low + self.min_bandwidth_freq)
        bandwidth_mel = np.linspace(hertz_to_mel(low_freq_hertz),
                                    hertz_to_mel(max_freq_hertz),
                                    self.out_channels+1)
        bandwidth_hertz = mel_to_hertz(bandwidth_mel)

        self.lower_freq_hertz = nn.parameter.Parameter(torch.tensor(bandwidth_hertz[:-1],dtype=torch.float)).reshape(-1,1)
        self.bandwidth_hertz = nn.parameter.Parameter(torch.tensor(np.diff(bandwidth_hertz),dtype=torch.float)).reshape(-1,1)

        N = torch.linspace(0,(self.kernel_size/2)-1,self.kernel_size//2)
        self.hanning_window = 0.54 - 0.46*torch.cos(2*torch.pi*N/self.kernel_size)
        self.two_pi_n = 2*torch.pi*torch.arange(-1*(self.kernel_size-1)/2,0,dtype=torch.float)/self.sampling_rate
        self.two_pi_n = self.two_pi_n.unsqueeze(0)


    def forward(
        self,
        sample
    ):
        self.two_pi_n = self.two_pi_n.to(sample.device)
        self.hanning_window = self.hanning_window.to(sample.device)

        f1_cutoff_freq = self.freq_low + torch.abs(self.lower_freq_hertz)
        f2_cutoff_freq = torch.clamp(f1_cutoff_freq+self.min_bandwidth_freq+torch.abs(self.bandwidth_hertz),
                                        self.freq_low,self.sampling_rate/2)
        bandwidth = (f2_cutoff_freq - f1_cutoff_freq)[:,0]
        
        f1_cutoff_freq = torch.matmul(f1_cutoff_freq,self.two_pi_n)
        f2_cutoff_freq = torch.matmul(f2_cutoff_freq,self.two_pi_n)

        band_pass_left = (torch.sin(f2_cutoff_freq) - torch.sin(f1_cutoff_freq))/(torch.div(self.two_pi_n,2,rounding_mode="floor"))
        band_pass_left *= self.hanning_window
        centre =  torch.unsqueeze((f2_cutoff_freq-f1_cutoff_freq)[:,0],1)

        band_pass = torch.cat([band_pass_left,
                              centre,
                              torch.flip(band_pass_left,dims=[1])
                             ],dim=1) / (2*bandwidth.unsqueeze(1))
        
        self.filters = band_pass.reshape(self.out_channels,1,self.kernel_size)
        
        return conv1d(sample,self.filters,padding=self.padding,
                        stride=self.stride,dilation=self.dilation,
                        bias=None, groups=1)

class SincNet(nn.Module):

    def __init__(
        self,
    ):
        super().__init__()

        self.conv1d = nn.ModuleList()
        self.pool1d = nn.ModuleList()
        self.layernorm1d = nn.ModuleList()
        self.dropout1d = nn.ModuleList()

        self.conv1d.append(
            SincConv(out_channels=80,
            kernel_size=251)
        )
        self.pool1d.append(nn.MaxPool1d(3,stride=3,padding=0,dilation=1))
        self.layernorm1d.append(nn.InstanceNorm1d(80,affine=True))
        self.dropout1d.append(nn.Dropout(0.1))
        
        self.conv1d.append(nn.Conv1d(80, 60, 5, stride=1))
        self.pool1d.append(nn.MaxPool1d(3, stride=3, padding=0, dilation=1))
        self.layernorm1d.append(nn.InstanceNorm1d(60, affine=True))
        self.dropout1d.append(nn.Dropout(0.1))


        self.conv1d.append(nn.Conv1d(60, 60, 5, stride=1))
        self.pool1d.append(nn.MaxPool1d(3, stride=3, padding=0, dilation=1))
        self.layernorm1d.append(nn.InstanceNorm1d(60, affine=True))
        self.dropout1d.append(nn.Dropout(0.1))
        self.activation = nn.LeakyReLU()

    def forward(
        self,
        sample
    ):
        output = sample ##change
        for i,(conv,pool,norm,drop) in enumerate(
                                        zip(self.conv1d,self.pool1d,
                                        self.layernorm1d,self.dropout1d)
        ):
            output = conv(output)
            if i == 0:
                output = torch.abs(output)

            output = drop(self.activation(norm(pool(output))))
        
        return output

class SegmentNet(nn.Module):

    def __init__(
        self,
        
    ):
        self.sincnet = SincNet()
        self.lstm = nn.LSTM(input_size=60, hidden_size=128, num_layers=4, bidirectional=True, dropout=0.0)
        self.classifier = nn.ModuleList(128*2,4)
        self.activation = nn.Sigmoid()

    def forward(
        self,
        sample
    ):
        output = self.sincnet(sample)
        output,hidden = self.lstm(output)
        output = self.activation(self.classifier(output))

        return output


