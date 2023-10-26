import torch
import torch.nn as nn
from tqdm import tqdm

def train(epoch, train_loader, generator, discriminator, optimizer_G, optimizer_D, criterion):
    generator.train()
    discriminator.train()
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

    G_losses = []
    D_losses = []

    D_accuracies = []

    for music, image in pbar:

        music = music.to(device)
        image = image.to(device)

        # Train discriminator on real images
        optimizer_D.zero_grad()
        D_real_output = discriminator(image)
        D_real_loss = criterion(D_real_output, real_label)

        # Train discriminator on fake images
        noise = torch.randn(BATCH_SIZE, dim).to(device) # dim - image dimension
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

        pbar.set_postfix(f'\n\tGenerator Loss: {np.mean(G_losses):.5f}\n\tDiscriminator Loss: {np.mean(D_losses):.5f}\n\tDiscriminator Accuracy: {np.mean(D_accuracies):.5f}')