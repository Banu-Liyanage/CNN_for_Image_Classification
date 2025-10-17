import torch
from cnn_model import SimpleCNN

# Instantiate model
model = SimpleCNN(num_classes=9)

# Create a batch of 4 fake RGB images, 64x64 each
sample_input = torch.randn(4, 3, 64, 64)

# Forward pass through the model
output = model(sample_input)

print("Model executed successfully!")
print("Input shape :", sample_input.shape)#if this is printed model is working fine
print("Output shape:", output.shape)
print("Output tensor:\n", output)
