# Image-Classification-Captioning

# VisionTransformer-Captioning  

The project (that I completed for my CMPUT 328 class) focused on two tasks:  

1. **Classification with Vision Transformer (ViT)** on the CIFAR-10 dataset.  
2. **Image Captioning** using Vision Transformer and GPT-2 on the Flickr8k dataset.  

Both tasks highlight the application of state-of-the-art transformer models in vision-based machine learning tasks.  

---

## Key Features  

- **Task 1**: Implemented and trained a Vision Transformer (ViT) model, achieving a test accuracy of over 65% on CIFAR-10.  
- **Task 2**: Developed a sequence-to-sequence image captioning model combining a pretrained ViT as the encoder and GPT-2 as the decoder. The model achieved a BLEU score exceeding 0.07%.  

---

## Repository Structure  
```
VisionTransformer-Captioning
├── vit_submission.py           # Completed implementation for ViT classification.
├── vit-cifar10.pt              # Trained ViT model on CIFAR-10.
├── cap_submission.py           # Completed implementation for image captioning.
├── cap-vlm.pt                  # Trained image captioning model.
└── README.md                   # Project overview and usage instructions.
```
---

## How to Use  

1. **Classification with ViT**  
   - Run `vit_main.py` with the trained model `vit-cifar10.pt` to evaluate classification performance.  

2. **Image Captioning**  
   - Run `cap_main.py` with the trained model `cap-vlm.pt` to generate captions for input images.  

---

## Results  

- **Task 1**: The ViT model achieved a test accuracy exceeding the 65%.  
- **Task 2**: The image captioning model got the BLEU score of 0.07% on the validation dataset, ensuring readiness for the hidden test set.  

#### Example 1  
![Example 1](result_1.jpg)  
*Caption generated: "A little girl in a pink bathing suit is splashing in a sprinkler. "*

#### Example 2  
![Example 2](result_2.jpg)  
*Caption generated: "A little girl in a pink dress is standing in a field of flowers."*

#### Example 3  
![Example 3](result_3.jpg)  
*Caption: "A child in a red jacket is shoveling snow."*  
