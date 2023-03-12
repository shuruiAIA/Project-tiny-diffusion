import numpy as np

import torch
from torch import nn

import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MedianBlur(nn.Module):
    def __init__(self, network, num_timesteps, num_timesteps_generate, kernel_size ,device=device) -> None:
        super(MedianBlur, self).__init__()
        self.num_timesteps = num_timesteps
        self.num_timesteps_generate = num_timesteps_generate
        self.network = network.to(device)
        self.kernel_size = kernel_size

    def forward_process(self, x, t_tensor):
        # The forward process, we add median blur for the input image
        res = torch.zeros_like(x)
        for i in range(x.shape[0]):
            res_numpy = x[i].squeeze().cpu().numpy()
            for j in range(t_tensor[i].item()):
                res_numpy = cv2.medianBlur(res_numpy, self.kernel_size)
            res[i] = torch.from_numpy(res_numpy).unsqueeze(0)
        return res

    def reverse(self, x, t):
        # The network return the image we estimated before the blur

        return self.network.forward(x, t)

    # two ways to restore the images
    def restore_step_algo1(self, x, t_tensor):
        x_0 = self.reverse(x, t_tensor)
        
        return self.forward_process(x_0, t_tensor-1)
    
    def restore_step_algo2(self, x, t_tensor):
        x_0 = self.reverse(x, t_tensor)
        
        return x - self.forward_process(x_0, t_tensor) + self.forward_process(x_0, t_tensor-1)
    

class ConvolutionBlur(nn.Module):
    # Here we can use the kernel of the convolution to add blur for the image
    def __init__(self, network, num_timesteps, num_timesteps_generate, kernel, device=device) -> None:
        super(ConvolutionBlur, self).__init__()
        self.num_timesteps = num_timesteps
        self.num_timesteps_generate = num_timesteps_generate
        self.network = network.to(device)
        self.kernel = kernel

    def forward_process(self, x, t_tensor):
        # The forward process, we add blur for the input image
        res = torch.zeros_like(x)
        for i in range(x.shape[0]):
            res_numpy = x[i].squeeze().cpu().numpy()
            for j in range(t_tensor[i].item()):
                res_numpy = cv2.filter2D(res_numpy, -1, self.kernel)
            res[i] = torch.from_numpy(res_numpy).unsqueeze(0)
        return res
    

    def reverse(self, x, t):
        # The network return the image we estimated before the blur

        return self.network.forward(x, t)

    # two ways to restore the images
    def restore_step_algo1(self, x, t_tensor):
        x_0 = self.reverse(x, t_tensor)
        
        return self.forward_process(x_0, t_tensor-1)
    
    def restore_step_algo2(self, x, t_tensor):
        x_0 = self.reverse(x, t_tensor)
        
        return x - self.forward_process(x_0, t_tensor) + self.forward_process(x_0, t_tensor-1)
    

class SuperResolution(nn.Module):
    def __init__(self, network, num_timesteps, device=device) -> None:
        super(SuperResolution, self).__init__()
        self.num_timesteps = num_timesteps
        self.num_timesteps_generate = num_timesteps
        self.network = network.to(device)

    def forward_process(self, x, t_tensor):
        # The forward process
        res = torch.zeros_like(x)
        for i in range(x.shape[0]):
            res_numpy = x[i].squeeze().cpu().numpy().astype(np.uint8)
            r = t_tensor[i].item()
            mask = cv2.circle(np.ones(res_numpy.shape[:2],dtype=np.uint8), (14, 14), r, 0, -1)
            res_numpy = cv2.add(res_numpy, np.zeros(np.shape(res_numpy),dtype=np.uint8), mask=mask)
            res[i] = torch.from_numpy(res_numpy).unsqueeze(0)
            
        return res

    def reverse(self, x, t):
        # The network return the image we estimated before the blur

        return self.network.forward(x, t)
    
    def restore_step_algo1(self, x, t_tensor):
        x_0 = self.reverse(x, t_tensor)
        
        return self.forward_process(x_0, t_tensor-1)
    
    def restore_step_algo2(self, x, t_tensor):
        x_0 = self.reverse(x, t_tensor)
        
        return x - self.forward_process(x_0, t_tensor) + self.forward_process(x_0, t_tensor-1)