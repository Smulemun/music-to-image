class Music2ImageGenerator(nn.Module):
    def __init__(self, embedding_dim= 512):
        super().__init__()
        self.embedding = wav2clip.get_model()
        self.test = nn.ConvTranspose2d(embedding_dim, embedding_dim // 2, kernel_size=10, stride=6, padding=1),
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, embedding_dim // 2, kernel_size=10, stride=6, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(embedding_dim // 2, embedding_dim // 4, kernel_size=10, stride=6, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(embedding_dim // 4, 3, kernel_size=10, stride=6, padding=1),
            nn.Tanh(),
        )

    def forward(self, batch_audio):
        embedded = torch.stack([torch.tensor(wav2clip.embed_audio(audio, self.embedding)).reshape((-1, )) for audio in batch_audio])
        embedded = embedded.view(embedded.size(0), embedded.size(1), 1, 1)
        print(embedded[0].shape)
        print(self.test(embedded[0]).shape)
        return self.deconv(embedded)