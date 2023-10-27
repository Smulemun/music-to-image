import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

def evaluate(epoch, val_loader, generator, discriminator, criterion, device, real_label, fake_label):
    generator.eval()
    discriminator.eval()

    G_losses = []
    D_losses = []

    D_accuracies = []

    pbar = tqdm(val_loader)
    with torch.no_grad():
        for music, image in pbar:

            music = music.to(device)
            image = image.to(device).to(torch.float32) # !!!!!!!

            # Evaluate discriminator on real images
            D_real_output = discriminator(image)
            D_real_loss = criterion(D_real_output, real_label)

            # Evaluate discriminator on fake images
            noise = torch.randn(music.size(0), 512).to(device)
            fake_images = generator(noise, music)
            D_fake_output = discriminator(fake_images.detach())

            D_fake_loss = criterion(D_fake_output, fake_label)

            discriminator_loss = D_real_loss + D_fake_loss
            D_losses.append(discriminator_loss.item())

            D_accuracy = np.mean(D_real_output.cpu().detach().numpy() > 0.5)
            D_accuracies.append(D_accuracy)

            # Evaluate generator
            D_fake_output = discriminator(fake_images)
            G_loss = criterion(D_fake_output, real_label)
            G_losses.append(G_loss.item())

            pbar.set_description(f'\tValidation Epoch {epoch}, Generator Loss: {np.mean(G_losses):.5f}, Discriminator Loss: {np.mean(D_losses):.5f}, Discriminator Accuracy: {np.mean(D_accuracies):.5f}')

    return np.mean(G_losses), np.mean(D_losses)