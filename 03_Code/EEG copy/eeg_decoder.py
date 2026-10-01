
import torch
import torch.nn as nn

class EEGDecoder(nn.Module):
    """
    A lightweight CNN-based EEG decoder for gesture classification.
    It expects input of shape (B, 8, 1250) and outputs logits for 7 classes.
    """
    def __init__(self, num_classes=7, dropout_rate=0.5):
        super(EEGDecoder, self).__init__()

        # Block 1: Temporal Feature Extractor
        # Input: (B, 1, 8, 1250)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 50), stride=(1, 1), padding='same')
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))

        # Block 2: Spatial Feature Extractor
        # Input: (B, 32, 8, 312)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(8, 1), stride=(1, 1), padding='valid')
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        # Block 3: Deeper Temporal Features
        # Input: (B, 64, 1, 156)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 10), stride=(1, 1), padding='same')
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        # Adaptive Pooling to flatten features irrespective of input length
        # Input: (B, 128, 1, 78)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Flatten layer
        # Input: (B, 128, 1, 1)
        self.flatten = nn.Flatten()

        # Classifier Head
        # Input: (B, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        # Input x expected shape: (B, 8, 1250)
        # Add a channel dimension for Conv2d: (B, 1, 8, 1250)
        x = x.unsqueeze(1)

        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Adaptive Pooling and Flatten
        x = self.adaptive_pool(x)
        x = self.flatten(x)

        # Classifier Head
        x = self.dropout(x)
        x = self.classifier(x)

        return x
