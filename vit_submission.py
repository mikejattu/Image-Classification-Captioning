from typing import Tuple
import datetime

import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Args:
    """TODO: Command-line arguments to store model configuration.
    """
    num_classes = 10

    # Hyperparameters
    epochs = 25     # Should easily reach above 65% test acc after 20 epochs with an hidden_size of 64
    batch_size = 32
    lr = 0.001
    weight_decay = 0.0001

    # TODO: Hyperparameters for ViT
    # Adjust as you see fit
    input_resolution = 32
    in_channels = 3
    patch_size = 3
    hidden_size = 64
    layers = 6
    heads = 8


    # Save your model as "vit-cifar10-{YOUR_CCID}"
    YOUR_CCID = "sjattu"
    name = f"vit-cifar10-{YOUR_CCID}"

class PatchEmbeddings(nn.Module):
    """TODO: (0.5 out of 10) Compute patch embedding
    of shape `(batch_size, seq_length, hidden_size)`.
    """
    def __init__(
        self, 
        input_resolution: int,
        patch_size: int,
        hidden_size: int,
        in_channels: int = 3,      # 3 for RGB, 1 for Grayscale
        ):
        super().__init__()
        # #########################
        # finding the number of patches
        self.num_patches = ( input_resolution// patch_size ) ** 2
        # this is the patch embedding
        self.projection = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        # #########################
        

    def forward(
        self, 
        x: torch.Tensor,
        ) -> torch.Tensor:

        # #########################
        # passing it through the projection layer
        x = self.projection(x)
        # reshaping the tensor
        # flatten to : (batch_size, hidden_size, num_patches_h, num_patches_w) ---> (batch_size, hidden_size, num_patches)  
        # transpose to : (batch_size, hidden_size, num_patches) ---> (batch_size, num_patches, hidden_size)
        x = x.flatten(2).transpose(1, 2)
        # #########################

        embeddings = x

        # #########################
        return embeddings

class PositionEmbedding(nn.Module):
    def __init__(
        self,
        num_patches: int,
        hidden_size: int,
        ):
        """TODO: (0.5 out of 10) Given patch embeddings, 
        calculate position embeddings with [CLS] and [POS].
        """
        super().__init__()
        # #########################
        # Initializing the cls token and position embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, hidden_size))
        # #########################

    def forward(
        self,
        embeddings: torch.Tensor
        ) -> torch.Tensor:

        # #########################
        # expanding the cls token to the batch size
        cls_token = self.cls_token.expand(embeddings.shape[0], -1, -1)
        # concatenating the cls token to the embeddings
        embeddings = torch.cat([cls_token, embeddings], dim=1)
        # adding the position embeddings to the embeddings
        embeddings += self.position_embeddings
        # #########################

        return embeddings


class TransformerEncoderBlock(nn.Module):
    """TODO: (0.5 out of 10) A residual Transformer encoder block.
    """
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        # #########################
        
        # Defining the depth of the feedforward network
        self.dim_feedforward = 4 * d_model
        # Multihead attention layer
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        # Layer normalization layer 1
        self.ln_1 = nn.LayerNorm(d_model)
        # Multi-layer perceptron
        self.mlp = nn.Sequential(
            nn.Linear(d_model, self.dim_feedforward),
            nn.GELU(),
            nn.Linear(self.dim_feedforward, d_model)
        )
        # Layer normalization layer 2
        self.ln_2 = nn.LayerNorm(d_model)

        # #########################

    def forward(self, x: torch.Tensor):
        # #########################

        x_1 = self.ln_1(x)  # LayerNorm
        x_1 = self.attn(x_1, x_1, x_1)[0] # Multihead Attention
        x = x + x_1  # Residual connection

        x_1 = self.ln_2(x)  # LayerNorm
        x_1 = self.mlp(x_1)  # Feedforward
        x = x + x_1 # Residual connection
        
        # #########################

        return x


class ViT(nn.Module):
    """TODO: (0.5 out of 10) Vision Transformer.
    """
    def __init__(
        self, 
        num_classes: int,
        input_resolution: int, 
        patch_size: int, 
        in_channels: int,
        hidden_size: int, 
        layers: int, 
        heads: int,
        ):
        super().__init__()
        self.hidden_size = hidden_size
        # #########################
        # Finish Your Code HERE
        # #########################
        self.layers = nn.ModuleList([TransformerEncoderBlock(hidden_size, heads) for _ in range(layers)])
        self.patch_embed = PatchEmbeddings(input_resolution, patch_size, hidden_size, in_channels)
        self.pos_embed = PositionEmbedding(self.patch_embed.num_patches, hidden_size)
        self.ln_pre = nn.LayerNorm(hidden_size)
        self.transformer = nn.Sequential(*self.layers)
        self.ln_post = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

        # #########################


    def forward(self, x: torch.Tensor):
        # #########################
        x = self.patch_embed(x) # Patch Embedding
        x = self.pos_embed(x)   # Position Embedding
        x = self.ln_pre(x)      # LayerNorm

        x = self.transformer(x) # Transformer Encoder Block
        CLS = x[:, 0]             # Extract [CLS] token       
        x = self.ln_post(CLS)     # LayerNorm
        x = self.classifier(x)  # Classifier
    
        # #########################

        return x


def transform(
    input_resolution: int,
    mode: str = "train",
    mean: Tuple[float] = (0.5, 0.5, 0.5),   # NOTE: Modify this as you see fit
    std: Tuple[float] = (0.5, 0.5, 0.5),    # NOTE: Modify this as you see fit
    ):
    """TODO: (0.25 out of 10) Preprocess the image inputs
    with at least 3 data augmentation for training.
    """
    if mode == "train":
        # #########################

        tfm = transforms.Compose([
            # Data Augmentation 1: Random Horizontal Flip
            transforms.RandomHorizontalFlip(),
            # Data Augmentation 2: Random Rotation
            transforms.RandomRotation(10),
            # Data Augmentation 3: Random Vertical Flip
            transforms.RandomVerticalFlip(),

            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        # #########################

    else:
        # #########################

        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # #########################

    return tfm

def inverse_transform(
    img_tensor: torch.Tensor,
    mean: Tuple[float] = (-0.5/0.5, -0.5/0.5, -0.5/0.5),    # NOTE: Modify this as you see fit
    std: Tuple[float] = (1/0.5, 1/0.5, 1/0.5),              # NOTE: Modify this as you see fit
    ) -> np.ndarray:
    """Given a preprocessed image tensor, revert the normalization process and
    convert the tensor back to a numpy image.
    """
    # #########################
    # Finish Your Code HERE
    # #########################
    inv_normalize = transforms.Normalize(mean=mean, std=std)
    img_tensor = inv_normalize(img_tensor).permute(1, 2, 0)
    img = np.uint8(255 * img_tensor.numpy())
    # #########################
    return img


def train_vit_model(args):
    """TODO: (0.25 out of 10) Train loop for ViT model.
    """
    # #########################
    # -----
    # Dataset for train / test
    tfm_train = transform(
        input_resolution=args.input_resolution, 
        mode="train",
    )

    tfm_test = transform(
        input_resolution=args.input_resolution, 
        mode="test",
    )
    # #########################
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=tfm_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=tfm_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # -----
    # TODO: Define ViT model here
    model = ViT(\
        num_classes=args.num_classes, 
        input_resolution=args.input_resolution, 
        patch_size=args.patch_size, 
        in_channels=args.in_channels, 
        hidden_size=args.hidden_size, 
        layers=args.layers, 
        heads=args.heads)
    # print(model)

    if torch.cuda.is_available():
        model.cuda()

    # TODO: Define loss, optimizer and lr scheduler here

    # Defining the loss function as CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()

    # Defining the optimizer as Adam
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

    # Defining the learning rate scheduler as ReduceLROnPlateau
    scheduler =  ReduceLROnPlateau(optimizer, 'max', factor = 0.01, patience = 3, threshold= 0.01, threshold_mode='abs')

    # #########################
    # Evaluate at the end of each epoch
    best_acc = 0.0
    for epoch in range(args.epochs):
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1} / {args.epochs}")

        for i, (x, labels) in enumerate(pbar):
            model.train()
            # #########################
            # Finish Your Code HERE
            # #########################
            # Forward pass
            if torch.cuda.is_available():
                x = x.cuda()
                labels = labels.cuda()
            # Zeroing the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(x)
            # Compute loss
            loss = criterion(outputs, labels)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            # #########################

            # NOTE: Show train loss at the end of epoch
            # Feel free to modify this to log more steps
            pbar.set_postfix({'loss': '{:.4f}'.format(loss.item())})

        # Evaluate at the end
        test_acc = test_classification_model(model, test_loader)
        # changing the learning rate with respect to the test accuracy 
        scheduler.step(test_acc)
        # NOTE: DO NOT CHANGE
        # Save the model
        if test_acc > best_acc:
            best_acc = test_acc
            state_dict = {
                "model": model.state_dict(),
                "acc": best_acc,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            torch.save(state_dict, "{}.pt".format(args.name))
            print("Best test acc:", best_acc)
        else:
            print("Test acc:", test_acc)
        print()

def test_classification_model(
    model: nn.Module,
    test_loader,
    ):
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total
