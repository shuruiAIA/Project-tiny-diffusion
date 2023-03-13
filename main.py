import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F
import functools
import torchvision
from torchvision import transforms
from tqdm.auto import tqdm

from UNet import MyUNet, MyTinyUNet
from UNet_conditional import UNet_conditional
from ddpm import DDPM
from ddpm_cold import MedianBlur, ConvolutionBlur, SuperResolution
from score_utils import AverageMeter,AnnealedLangevinDynamic
from score_model import Score_Model
from score_sde_model import Score_SDE_Model
from score_sde_utils import pc_sampler,EMA,marginal_prob_std,diffusion_coeff,loss_fn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def training_loop(model, dataloader, optimizer, num_epochs, num_timesteps, device=device):
    """Training loop for DDPM"""

    global_step = 0
    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(dataloader):
            batch = batch[0].to(device)
            noise = torch.randn(batch.shape).to(device)
            timesteps = torch.randint(0, num_timesteps, (batch.shape[0],)).long().to(device)

            noisy = model.add_noise(batch, noise, timesteps)
            noise_pred = model.reverse(noisy, timesteps)
            loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step}
            losses.append(loss.detach().item())
            progress_bar.set_postfix(**logs)
            global_step += 1
        progress_bar.close()

def training_loop_conditional(model, dataloader, optimizer, num_epochs, num_timesteps, nb_classes=10, device=device):
    global_step = 0
    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        for step, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            noise = torch.randn(images.shape).to(device)
            timesteps = torch.randint(0, num_timesteps, (images.shape[0],)).long().to(device)

            noisy = model.add_noise(images, noise, timesteps)
            noise_pred = model.reverse(noisy, timesteps, labels)
            loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step}
            losses.append(loss.detach().item())
            progress_bar.set_postfix(**logs)
            global_step += 1
        progress_bar.close()

def training_loop_cold(model, dataloader, optimizer, num_epochs, num_timesteps, device=device):
    """Training loop for cold diffusion"""

    global_step = 0
    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(dataloader):
            batch = batch[0].to(device)
   
            timesteps = torch.randint(1, num_timesteps, (batch.shape[0],)).long().to(device)

            blured = model.forward_process(batch, timesteps)
            image_pred = model.reverse(blured, timesteps)
            loss = F.mse_loss(image_pred, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step}
            losses.append(loss.detach().item())
            progress_bar.set_postfix(**logs)
            global_step += 1
        progress_bar.close()

def training_loop_score(model, dataloader, optimizer, total_iteration, device=device):
    losses= AverageMeter('Loss', ':.4f')
    while current_iteration != total_iteration:
        model.train()
        try:
            data = next(dataiterator)
        except:
            dataiterator = iter(dataloader)
            data = next(dataiterator)
        data = next(dataiterator)
        data = data[0].to(device = device)
        loss = model.loss_fn(data)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item())
        current_iteration += 1
        if current_iteration % 2000 == 0:
            losses.reset()

def training_loop_sde(model, dataloader, optimizer, n_epochs = 50, device=device):
    ema=EMA(model)
    tqdm_epoch = tqdm.notebook.trange(n_epochs)
    
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for x, y in dataloader:
            x = x.to(device)    
            loss = loss_fn(model, x, marginal_prob_std_fn)
            optimizer.zero_grad()
            loss.backward()    
            optimizer.step()
            ema.update(model)
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
        # Print the averaged training loss so far.
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
    

def generate_image(ddpm, sample_size, channel, height, width):
    """Generate the image from the Gaussian noise"""

    frames = []
    frames_mid = []
    with torch.no_grad():
        timesteps = list(range(ddpm.num_timesteps))[::-1]
        sample = torch.randn(sample_size, channel, height, width).to(device)
        
        for i, t in enumerate(tqdm(timesteps)):
            time_tensor = (torch.ones(sample_size, 1) * t).long().to(device)
            residual = ddpm.reverse(sample, time_tensor).to(device)
            sample = ddpm.step(residual, time_tensor[0], sample)

            if t==500:
                sample_squeezed = torch.squeeze(sample)
                for i in range(sample_size):
                    frames_mid.append(sample_squeezed[i].detach().cpu().numpy())

        sample = torch.squeeze(sample)
        for i in range(sample_size):
            frames.append(sample[i].detach().cpu().numpy())
    return frames, frames_mid

def generate_image_conditional(ddpm, labels, sample_size, channel, height, width):

    frames = []
    frames_mid = []
    with torch.no_grad():
        timesteps = list(range(ddpm.num_timesteps))[::-1]
        sample = torch.randn(sample_size, channel, height, width).to(device)
        
        for i, t in enumerate(tqdm(timesteps)):
            time_tensor = (torch.ones(sample_size, 1) * t).long().to(device)
            predicted_noise = ddpm.reverse(sample, time_tensor, labels).to(device)
            sample = ddpm.step(predicted_noise, time_tensor[0], sample)

            if t==500:
                sample_squeezed = torch.squeeze(sample)
                for i in range(sample_size):
                    frames_mid.append(sample_squeezed[i].detach().cpu().numpy())

        sample = torch.squeeze(sample)
        for i in range(sample_size):
            frames.append(sample[i].detach().cpu().numpy())
    return frames, frames_mid

def generate_image_cold(model, images, algoindex):
    """Generate the image from the final blured image"""

    frames = []
    sample_size = images.shape[0]
    with torch.no_grad():
        time_tensor = (torch.ones(sample_size) * model.num_timesteps).long().to(device)
        blured = model.forward_process(images, time_tensor).to(device)
    
        timesteps = list(range(1, model.num_timesteps))[::-1]
        
        for i, t in enumerate(tqdm(timesteps)):
            if t>1:
                time_tensor = (torch.ones(sample_size, 1) * t).long().to(device)
                if algoindex==1:
                    blured = model.restore_step_algo1(blured, time_tensor)
                if algoindex==2:
                    blured = model.restore_step_algo2(blured, time_tensor)
            else:
                time_tensor = (torch.ones(sample_size, 1) * t).long().to(device)
                blured = model.reverse(blured, time_tensor)

        for i in range(sample_size):
            frames.append(blured[i].detach().cpu().numpy())
    return frames

def show_images(images, title="",pixel=28):
    """Shows the provided images as sub-pictures in a square"""

    # Converting images to CPU numpy arrays
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx].reshape(pixel, pixel, 1), cmap="gray")
                idx += 1
    fig.suptitle(title, fontsize=30)
    # Showing the figure
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="ddpm", help="ddpm/ddpm_conditional/cold_median/cold_kernel/cold_resolution/score/sde")
    config = parser.parse_args()

    learning_rate = 1e-3
    dataset = torchvision.datasets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=128, shuffle=True)

    if config.experiment_name == "ddpm":
        num_epochs = 200
        num_timesteps = 1000
        network = MyTinyUNet()
        model = DDPM(network, num_timesteps, beta_start=0.0001, beta_end=0.02, device=device)
        optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate)
        training_loop(model, dataloader, optimizer, num_epochs, num_timesteps, device=device)
        generated, generated_mid = generate_image(model, 100, 1, 28, 28)
        show_images(generated_mid, "Mid result")
        show_images(generated, "Final result")

    if config.experiment_name == "ddpm_conditional":
        num_epochs = 200
        num_timesteps = 1000
        network = UNet_conditional()
        model = DDPM(network, num_timesteps, beta_start=0.0001, beta_end=0.02, device=device)
        optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate)
        training_loop(model, dataloader, optimizer, num_epochs, num_timesteps, device=device)
        labels = torch.arange(10).long().to(device).repeat(10)
        generated, generated_mid = generate_image(model, 100, 1, 28, 28)
        show_images(generated_mid, "Mid result")
        show_images(generated, "Final result")

    if config.experiment_name == "cold_median":
        num_epochs = 100
        num_timesteps = 100
        num_timesteps_generate = 100
        kernel_size = 3

        network = MyUNet()
        model = MedianBlur(network, num_timesteps, num_timesteps_generate, kernel_size, device=device)
        optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate)
        training_loop_cold(model, dataloader, optimizer, num_epochs, num_timesteps, device=device)

        images = dataset.data[0:100].unsqueeze(1)
        generated_1 = generate_image(model, images, 1)
        generated_2 = generate_image(model, images, 2)
        time_tensor = (torch.ones(100, 1) * model.num_timesteps).long()
        blured_final = model.forward_process(images, time_tensor)
        show_images(generated_1, "Final result1")
        show_images(generated_2, "Final result2")
        show_images(blured_final, "Final blured")


    if config.experiment_name == "cold_kernel":
        num_epochs = 50
        num_timesteps = 30
        num_timesteps_generate = 25
        # Here we can choose the kernel
        kernel = 1/9*np.array([[1, 1, 1],[1, 1, 1],[1, 1, 1]]) # Mean kernel
        # kernel = 1/16*np.array([[1, 2, 1],[2, 4, 2],[1, 2, 1]]) # Gaussian kernel

        network = MyTinyUNet()
        model = ConvolutionBlur(network, num_timesteps, num_timesteps_generate, kernel, device=device)
        optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate)
        training_loop_cold(model, dataloader, optimizer, num_epochs, num_timesteps, device=device)

        images = dataset.data[0:100].unsqueeze(1)
        generated_1 = generate_image(model, images, 1)
        generated_2 = generate_image(model, images, 2)
        time_tensor = (torch.ones(100, 1) * model.num_timesteps).long()
        blured_final = model.forward_process(images, time_tensor)
        show_images(generated_1, "Final result1")
        show_images(generated_2, "Final result2")
        show_images(blured_final, "Final blured")


    if config.experiment_name == "cold_resolution":
        num_epochs = 50
        num_timesteps = 8

        network = MyUNet()
        model = SuperResolution(network, num_timesteps, device=device)
        optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate)
        training_loop_cold(model, dataloader, optimizer, num_epochs, num_timesteps, device=device)

        images = dataset.data[0:100].unsqueeze(1)
        generated_1 = generate_image_cold(model, images, 1)
        generated_2 = generate_image_cold(model, images, 2)
        time_tensor = (torch.ones(100, 1) * model.num_timesteps).long()
        blured_final = model.forward_process(images, time_tensor)
        show_images(generated_1, "Final result1")
        show_images(generated_2, "Final result2")
        show_images(blured_final, "Final blured")
    
    if config.experiment_name == "score":
        total_iteration = 10000
        current_iteration = 0
        sampling_number = 100
        only_final = True
        transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((14, 14)),
        torchvision.transforms.ToTensor()
        ])
        dataset = torchvision.datasets.MNIST(root = './MNIST', train=True, download=True, transform = transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = 256, drop_last = True)
        dataiterator = iter(dataloader)
        # epsilon of step size
        eps = 1.5e-5
        # sigma min and max of Langevin dynamic
        sigma_min = 0.005
        sigma_max = 10
        # Langevin step size and Annealed size
        n_steps = 10
        annealed_step = 100

        model = Score_Model(device,n_steps,sigma_min,sigma_max)
        optim = torch.optim.Adam(model.parameters(), lr = 0.005)
        training_loop_score(model, dataloader, optim, 30000,device)
        samplingMethod = AnnealedLangevinDynamic(sigma_min, sigma_max, n_steps, annealed_step, model, device, eps=eps)
        samples = samplingMethod.sampling(100, only_final)
        show_images(samples, "ScoreBased Model",pixel=14)

    if config.experiment_name == 'sde':
        dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
        marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=25.0)
        diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=25)
        sde_model = torch.nn.DataParallel(Score_SDE_Model(marginal_prob_std=marginal_prob_std_fn))
        sde_model = sde_model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        training_loop_sde(sde_model,dataloader,optimizer,50,device)
        samples = pc_sampler(sde_model,marginal_prob_std_fn,diffusion_coeff_fn,batch_size=64,num_steps=500,snr=0.16,device='cuda',eps=1e-3)
        show_images(samples, title="Score_SDE")


        

    