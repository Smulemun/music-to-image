import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

def train(epoch, train_loader, generator, discriminator, optimizer_G, optimizer_D, criterion, device, real_label, fake_label):
    generator.train()
    discriminator.train()
    pbar = tqdm(train_loader)

    G_losses = []
    D_losses = []

    D_accuracies = []

    for music, image in pbar:

        music = music.to(device)
        image = image.to(device).to(torch.float32) # !!!!!!!

        # Train discriminator on real images
        optimizer_D.zero_grad()
        D_real_output = discriminator(image)
        D_real_loss = criterion(D_real_output, real_label)

        # Train discriminator on fake images
        noise = torch.randn(music.size(0), 512).to(device) # dim - image dimension
        fake_images = generator(noise, music)
        D_fake_output = discriminator(fake_images.detach()) # detach to avoid training generator
        D_fake_loss = criterion(D_fake_output, fake_label)
        
        discriminator_loss = D_real_loss + D_fake_loss
        D_losses.append(discriminator_loss.item())

        D_accuracy = np.mean(D_real_output.cpu().detach().numpy() > 0.5)  
        D_accuracies.append(D_accuracy)

        discriminator_loss.backward()
        optimizer_D.step()
        
        # Train generator
        optimizer_G.zero_grad()
        D_fake_output = discriminator(fake_images)
        G_loss = criterion(D_fake_output, real_label)
        G_losses.append(G_loss.item())

        G_loss.backward()
        optimizer_G.step()

        pbar.set_description(f'Epoch {epoch}, Generator Loss: {np.mean(G_losses):.5f}, Discriminator Loss: {np.mean(D_losses):.5f}, Discriminator Accuracy: {np.mean(D_accuracies):.5f}')