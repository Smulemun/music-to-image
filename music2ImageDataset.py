from torch.utils.data import Dataset, DataLoader
import audio2numpy as a2n
import cv2
from tqdm import tqdm

IMAGE_PATH = 'data/images/'
MUSIC_PATH = 'data/music/'
IMAGE_SIZE = 256

class MusicImageDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
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
                music, sr = a2n.audio_from_file(music_path)
                image = self._readimg(image_path)
                self.data.append((music, image))
            except Exception as e:
                print(e)
                print(f'Image: {image_path}')
                print(f'Audio: {music_path}')
    def _readimg(self, image_path):
        image = cv2.imread(image_path)
        resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]