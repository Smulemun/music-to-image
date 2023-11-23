from torchvision import transforms
import numpy as np
import torch
import wav2clip as w2c
import audio2numpy as a2n

IMAGE_PATH = '../data/images/'
MUSIC_PATH = '../data/music/'
IMAGE_SIZE = 128


def postprocess_image(image):
    image = image.cpu()
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])
    if len(image.shape) == 4:
        image = image[0, :, :, :]
        
    return reverse_transforms(image)

def embed_audio(audio_path, model):
    audio, sr = a2n.audio_from_file(audio_path)
    if audio.shape[0] > 1:
        audio = audio[:, 0]
    audio_embedding = w2c.embed_audio(audio, model)
    audio_embedding = torch.tensor(np.squeeze(audio_embedding))
    return audio_embedding