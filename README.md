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
    <br />
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
        <li><a href="#gan-architecture">GAN architecture</a></li>
      </ul>
    </li>
     <li><a href="#license">License</a></li>
  </ol>
</details>

<!-- ABOUT -->
## About the project
The project's goal is to convert music into visual artwork using advanced techniques, enhancing the creative intersection of audio and visual domains. To achieve this, we use CLIP sound embeddings and a self-trained GAN as our image generator

<!-- MODEL -->
## Model Architecture
### CLIP sound embeddings
[CLIP](https://openai.com/research/clip) is a deep learning model developed by [OpenAI](https://openai.com/) that is capable of understanding and associating images and text in a semantically meaningful way. It leverages a powerful vision model and a language model trained jointly on a large corpus of text and images.

![](https://production-media.paperswithcode.com/methods/3d5d1009-6e3d-4570-8fd9-ee8f588003e7.png)

The critical idea behind CLIP is that semantically similar text and images are placed closer together in this shared high-dimensional space. As a result, you can compare the embeddings of text and images to measure their similarity or dissimilarity. CLIP embeddings enable a wide range of applications, such as image retrieval based on textual queries, zero-shot classification, and more, by understanding the relationships between textual and visual information.

### GAN architecture

![](https://miro.medium.com/v2/resize:fit:1400/1*yO9fLGCR9mOgTVWUKiYQSQ.png)

Generator Architecture:

The Music2ImageGenerator is designed to convert music and random noise into images. It comprises a sequence of five convolutional blocks. The first block takes a concatenated input of the random noise and music, reshapes it into a 4D tensor, and applies transposed convolution (also known as deconvolution) to upsample the data. The subsequent blocks continue this process, gradually reducing the spatial dimensions while increasing the number of channels. Leaky ReLU activation functions are applied after each convolutional layer to introduce non-linearity. The final block uses a hyperbolic tangent (tanh) activation function to output the generated image. Batch normalization is used in all blocks except the last one.

Discriminator Architecture:

The FakeImageDiscriminator is responsible for determining the authenticity of an image. It consists of five convolutional blocks followed by a fully connected layer. The convolutional layers progressively downsample the image while increasing the number of channels and apply ReLU activation functions. The final convolutional block uses a larger kernel size and no batch normalization. The output of the last convolutional block is flattened and fed into a fully connected layer to produce a single output value, which is passed through a sigmoid activation function to classify the image as real or fake. This architecture is commonly used in GANs for image discrimination tasks.


<!-- LICENSE -->
## License
Distributed under the MIT License. See `LICENSE` for more information.
