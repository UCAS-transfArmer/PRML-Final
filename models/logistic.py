import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    """
    A simple Logistic Regression model.
    Assumes input will be flattened.
    """
    def __init__(self, input_dim, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # Flatten the input tensor if it's not already flat (e.g., images)
        # Input x is expected to be [batch_size, channels, height, width] or [batch_size, features]
        if x.dim() > 2:
            x = x.view(x.size(0), -1) 
        out = self.linear(x)
        return out