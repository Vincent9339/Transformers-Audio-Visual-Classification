## Transformers Model for Audio/Visual Data Classification

This repository hosts a **Transformers-based model** for classifying musical instruments from audio and visual data, built using **Timm** and **PyTorch**. The project is inspired by and based on the paper *"Contrastive Audio-Visual Masked Autoencoder"* by **Gong et al. (2022)** and many more. All credit for the underlying concepts and methodologies goes to these works. You can access the main original paper here: [arXiv preprint arXiv:2210.07839](https://arxiv.org/abs/2210.07839).



### Key Features

- **Transformers Architecture**: Implements the latest advancements in Transformer models as outlined in *Gong et al. (2022)*, optimized for handling complex multimodal data.
- **Masked Modeling**: Uses masked modeling strategies to enhance feature learning, allowing the model to predict masked parts of the audio/visual inputs.
- **Dual Modality Learning**: Capable of processing both audio and visual inputs, enabling more robust classification by integrating cross-modality information.
- **Self-Supervised Learning**: Pre-training is done using unlabeled audio/visual data, leveraging the principles from the cited paper to learn representations without the need for large labeled datasets.
- **Supervised Fine-Tuning**: The model is then fine-tuned using labeled data for enhanced performance, achieving high accuracy in classifying a wide range of musical instruments.
- **Built with PyTorch & Timm**: Utilizes **PyTorch** along with the **Timm** library for efficient implementation, offering a flexible and scalable approach to model development.

### Applications

- **Music Recognition**: Classify musical instruments from audio or video recordings.
- **Multimodal Analysis**: Useful for analyzing audio-visual content in fields like music education, content creation, and sound synthesis.
- **Research and Development**: A resource for experimenting with recent advancements in Transformers for multimodal learning, as detailed in the *Gong et al. (2022)* study.

### Citation

> ```css
> @article{gong2022contrastive,
> title={Contrastive audio-visual masked autoencoder},
> author={Gong, Yuan and Rouditchenko, Andrew and Liu, Alexander H and Harwath, David and Karlinsky, Leonid and Kuehne, Hilde and Glass, James},
> journal={arXiv preprint arXiv:2210.07839},
> year={2022}
> }
> 
> @article{gong2021ast,
> title={Ast: Audio spectrogram transformer},
> author={Gong, Yuan and Chung, Yu-An and Glass, James},
> journal={arXiv preprint arXiv:2104.01778},
> year={2021}
> }
> 
> @inproceedings{he2022masked,
> title={Masked autoencoders are scalable vision learners},
> author={He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yanghao and Doll{\'a}r, Piotr and Girshick, Ross},
> booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
> pages={16000--16009},
> year={2022}
> }
>    
> Check the pdf for more
> ```
>
> 
