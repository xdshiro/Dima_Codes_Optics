"""
Module: classifier_models.py

Description:
    Defines various PyTorch classifier architectures for 3D and 2D data, as well as a flexible fully-connected network for spectral inputs.

    - Classifier3D: 3D convolutional neural network with configurable stages and pooling, followed by two fully-connected layers.
    - ClassifierFC_spec: Fully-connected network with batch normalization, dropout, and multiple configurable hidden layers for spectral data.
    - Classifier2D: 2D convolutional neural network with configurable stages and pooling, followed by two fully-connected layers.

Helper Functions:
    - conv_stage_2d: Build a 2D convolutional stage from a list of configurations.
    - create_pooling_layer_2d: Create a 2D max-pooling layer from a configuration tuple.
"""

import torch
from torch import nn

# -----------------------------------------------------------------------------
# 3D Convolutional Classifier
# -----------------------------------------------------------------------------
class Classifier3D(nn.Module):
    """
    3D convolutional classifier.

    Args:
        stages (list of list of tuples): Each stage is a list of (in_channels, out_channels, kernel_size, stride, padding).
        pooling_configs (list of tuples or None): Each entry is (kernel_size, stride, padding) or None for no pooling.
        num_classes (int): Number of output classes.
        desired_res (tuple of int): Input resolution (depth, height, width) for computing linear feature size.
    """
    def __init__(self, stages, pooling_configs, num_classes=11, desired_res=(32, 32, 32)):
        super(Classifier3D, self).__init__()
        # Sequential container for convolutional and pooling layers
        self.features = nn.Sequential()

        # Build convolutional stages and optional pooling
        for i, stage in enumerate(stages):
            self.features.add_module(f"stage_{i}", self.conv_stage(stage))
            if i < len(pooling_configs):
                pool_layer = self.create_pooling_layer(pooling_configs[i])
                if pool_layer:
                    self.features.add_module(f"pool_{i}", pool_layer)

        # Compute flattened feature size for linear layer
        self._to_linear = None
        self._get_conv_output((1, *desired_res))

        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def conv_stage(self, layer_configs):
        """
        Create a convolutional stage consisting of Conv3d, BatchNorm, and ReLU.
        """
        layers = []
        for config in layer_configs:
            in_ch, out_ch, k, s, p = config
            layers.append(nn.Conv3d(in_ch, out_ch, k, s, p))
            layers.append(nn.BatchNorm3d(out_ch))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def create_pooling_layer(self, config):
        """
        Create a 3D max-pooling layer or return None if config is None.
        """
        if config is None:
            return None
        k, s, p = config
        return nn.MaxPool3d(kernel_size=k, stride=s, padding=p)

    def _get_conv_output(self, shape):
        """
        Pass a dummy tensor through conv layers to compute the flattened feature size.
        """
        input_tensor = torch.rand(1, *shape)
        output_feat = self.features(input_tensor)
        # Multiply all dimensions except the batch
        self._to_linear = int(torch.prod(torch.tensor(output_feat.size()[1:])))

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor of shape (batch, channels, D, H, W).
        Returns:
            Tensor: Output logits of shape (batch, num_classes).
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# -----------------------------------------------------------------------------
# Flexible Fully-Connected Network for Spectral Data
# -----------------------------------------------------------------------------
class ClassifierFC_spec(nn.Module):
    """
    Fully-connected classifier with batch normalization and dropout.

    Args:
        input_size (int): Dimension of input features.
        hidden_sizes1 (int): Size of first hidden layer.
        hidden_sizes2 (int): Size of second (and repeated) hidden layers.
        hidden_sizes3 (int): Size of the third hidden layer before output.
        num_hidden (int): Number of repeated hidden layers with size hidden_sizes2.
        num_classes (int): Number of output classes.
        dropout_rate (float): Dropout probability.
    """
    def __init__(self, input_size, hidden_sizes1, hidden_sizes2, hidden_sizes3, num_hidden, num_classes, dropout_rate=0):
        super(ClassifierFC_spec, self).__init__()
        layers = []

        # Input -> hidden1 + BatchNorm + ReLU + Dropout
        layers.append(nn.Linear(input_size, hidden_sizes1))
        layers.append(nn.BatchNorm1d(hidden_sizes1))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout_rate))

        # hidden1 -> hidden2 + BatchNorm + ReLU + Dropout
        layers.append(nn.Linear(hidden_sizes1, hidden_sizes2))
        layers.append(nn.BatchNorm1d(hidden_sizes2))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout_rate))

        # Additional repeated hidden layers
        for _ in range(num_hidden):
            layers.append(nn.Linear(hidden_sizes2, hidden_sizes2))
            layers.append(nn.BatchNorm1d(hidden_sizes2))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))

        # hidden2 -> hidden3 + BatchNorm + ReLU + Dropout
        layers.append(nn.Linear(hidden_sizes2, hidden_sizes3))
        layers.append(nn.BatchNorm1d(hidden_sizes3))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout_rate))

        # Final output layer (no dropout)
        layers.append(nn.Linear(hidden_sizes3, num_classes))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        Forward pass through sequential linear layers.

        Args:
            x (Tensor): Input features of shape (batch, input_size).
        Returns:
            Tensor: Output logits of shape (batch, num_classes).
        """
        for layer in self.layers:
            x = layer(x)
        return x


# -----------------------------------------------------------------------------
# Helper Functions for 2D Convolutional Classifier
# -----------------------------------------------------------------------------
def conv_stage_2d(layer_configs):
    """
    Build a 2D convolutional stage: Conv2d -> BatchNorm -> ReLU.
    """
    layers = []
    for config in layer_configs:
        in_ch, out_ch, k, s, p = config
        layers.append(nn.Conv2d(in_ch, out_ch, k, s, p))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def create_pooling_layer_2d(config):
    """
    Create a 2D max-pooling layer or return None if config is None.
    """
    if config is None:
        return None
    k, s, p = config
    return nn.MaxPool2d(kernel_size=k, stride=s, padding=p)


# -----------------------------------------------------------------------------
# 2D Convolutional Classifier
# -----------------------------------------------------------------------------
class Classifier2D(nn.Module):
    """
    2D convolutional classifier.

    Args:
        stages (list of list of tuples): Each stage is a list of (in_channels, out_channels, kernel_size, stride, padding).
        pooling_configs (list of tuples or None): Each entry is (kernel_size, stride, padding) or None.
        num_classes (int): Number of output classes.
        shape_X_l (int): Height of input image.
        shape_X_p (int): Width of input image.
    """
    def __init__(self, stages, pooling_configs, num_classes=11, shape_X_l=7, shape_X_p=13):
        super(Classifier2D, self).__init__()
        self.shape_X_l = shape_X_l
        self.shape_X_p = shape_X_p

        # Sequential container for conv and pooling layers
        self.features = nn.Sequential()

        for i, stage in enumerate(stages):
            self.features.add_module(f"stage_{i}", conv_stage_2d(stage))
            if i < len(pooling_configs):
                pool_layer = create_pooling_layer_2d(pooling_configs[i])
                if pool_layer:
                    self.features.add_module(f"pool_{i}", pool_layer)

        # Compute flattened feature size for FC layers
        self._to_linear = None
        self._get_conv_output((1, shape_X_l, shape_X_p))

        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def _get_conv_output(self, shape):
        """
        Compute flattened feature size after conv and pooling with a dummy tensor.
        """
        input_tensor = torch.rand(1, *shape)
        output_feat = self.features(input_tensor)
        self._to_linear = int(torch.prod(torch.tensor(output_feat.size()[1:])))

    def initialize_weights(self):
        """
        Optional weight initialization method.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass for 2D classifier.

        Args:
            x (Tensor): Input tensor of shape (batch, shape_X_l * shape_X_p).
        Returns:
            Tensor: Output logits of shape (batch, num_classes).
        """
        batch_size = x.size(0)
        # Reshape flat input to image format with channel=1
        x = x.view(batch_size, 1, self.shape_X_l, self.shape_X_p)

        # Pass through convolutional layers
        x = self.features(x)

        # Flatten and pass through fully connected layers
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)