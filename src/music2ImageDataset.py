from torch.utils.data import Dataset
import audio2numpy as a2n
import cv2
from tqdm import tqdm
import wav2clip as w2c
import numpy as np

IMAGE_PATH = 'data/images/'
MUSIC_PATH = 'data/music/'
IMAGE_SIZE = 64

class MusicImageDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.w2c_model = w2c.get_model() 
        self._load()

    def _load(self):
        self.data = []
        pbar = tqdm(self.dataframe.iterrows(), total=len(self.dataframe), desc='Loading data')
        for _, row in pbar:
            music_id = row['music_id']
            image_id = row['image_id']
            music_path = MUSIC_PATH + music_id + '.wav'
            image_path = IMAGE_PATH + image_id + '.jpg'
            try:
                audio_embedding = self._read_audio(music_path)
                image = self._read_img(image_path)
                self.data.append((audio_embedding, image))
            except Exception as e:
                print(e)
                print(f'Image: {image_path}')
                print(f'Audio: {music_path}')

    def _read_img(self, image_path):
        image = cv2.imread(image_path)
        resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).reshape(3, IMAGE_SIZE, IMAGE_SIZE)

    def _read_audio(self, audio_path):
        audio, sr = a2n.audio_from_file(audio_path)
        audio_embedding = w2c.embed_audio(audio, self.w2c_model)
        audio_embedding = np.squeeze(audio_embedding)
        return audio_embedding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]