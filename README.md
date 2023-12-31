![Static Badge](https://img.shields.io/badge/CONTRIBUTORS-3-red?link=https%3A%2F%2Fgithub.com%2FSmulemun%2Fmusic-to-image%2Fgraphs%2Fcontributors)
![Static Badge](https://img.shields.io/badge/LICENSE-MIT-green?link=https%3A%2F%2Fgithub.com%2FSmulemun%2Fmusic-to-image%2Fblob%2Fmain%2FLICENSE)


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="https://static.vecteezy.com/system/resources/thumbnails/019/899/972/small/music-notes-white-free-png.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Model for music-to-image generation</h3>

  <p align="center">
    Creating art from music through generative AI.
    <br />
    <a href="https://github.com/Smulemun/music-to-image"><strong>Explore the docs »</strong></a>
     ·
    <a href="https://music2image.streamlit.app/"><strong>View Demo »</strong></a>
    <br />
  </p>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#model-architecture">Model Architecture</a>
      <ul>
        <li><a href="#clip-sound-embeddings">CLIP sound embeddings</a></li>
        <li><a href="#wav2clip">wav2CLIP</a></li>
        <li>
          <a href="#diffusion-model-architecture">Diffusion Model Architecture</a>
          <ul>
            <li><a href="#forward-process">Forward process</a></li>
            <li><a href="#unet-backward-process">Unet backward process</a></li>
          </ul>
        </li>
      </ul>
    </li>
     <li><a href="#license">License</a></li>
  </ol>
</details>

<!-- ABOUT -->
## About the project
The project's goal is to convert music into visual artwork using advanced techniques, enhancing the creative intersection of audio and visual domains. To achieve this, we use CLIP sound embeddings and a self-trained Diffusion Model as our image generator

<!-- MODEL -->
## Model Architecture

### CLIP sound embeddings
[CLIP](https://openai.com/research/clip) is a deep learning model developed by [OpenAI](https://openai.com/) that is capable of understanding and associating images and text in a semantically meaningful way. It leverages a powerful vision model and a language model trained jointly on a large corpus of text and images.

![](https://production-media.paperswithcode.com/methods/3d5d1009-6e3d-4570-8fd9-ee8f588003e7.png)

The critical idea behind CLIP is that semantically similar text and images are placed closer together in this shared high-dimensional space. As a result, you can compare the embeddings of text and images to measure their similarity or dissimilarity. CLIP embeddings enable a wide range of applications, such as image retrieval based on textual queries, zero-shot classification, and more, by understanding the relationships between textual and visual information.

### wav2CLIP 

[wav2CLIP](https://github.com/descriptinc/lyrebird-wav2clip) is a robust audio representation learning method by distilling from Contrastive Language-Image Pre-training (CLIP). We use this library to build music CLIP embeddings for our dataset.

### Diffusion Model Architecture

![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/latent-diffusion-arch.png)

#### Forward process:

Given a data point sampled from a real data distribution, let us define a forward diffusion process in which we add small amount of Gaussian noise to the sample in T steps, producing a sequence of noisy samples. The step sizes are controlled by a variance schedule 


![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/DDPM.png)


#### Unet backward process:

This is a U-Net based model to predict noise

U-Net is a gets it's name from the U shape in the model diagram. It processes a given image by progressively lowering (halving) the feature map resolution and then increasing the resolution. There are pass-through connection at each resolution.

![](https://nn.labml.ai/unet/unet.png)

<!-- LICENSE -->
## License
Distributed under the MIT License. See `LICENSE` for more information.
