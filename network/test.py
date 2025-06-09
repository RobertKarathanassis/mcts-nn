import torch
from network import JohnNet

# Create the model
model = JohnNet()

# Create a dummy input: batch of 2 boards
dummy_input = torch.randn(2, 20, 8, 8)

# Forward pass
policy_logits, value = model(dummy_input)

# Print shapes
print("Policy shape:", policy_logits.shape)  # Expect (2, 4672)
print("Value shape:", value.shape)          # Expect (2, 1)

# Check value range
print("Value output:", value)  # Should be in [-1, 1]

# Try backprop
dummy_target = torch.tensor([[0.5], [-0.2]])  # Random targets
loss = torch.nn.functional.mse_loss(value, dummy_target)
loss.backward()

print("Gradients OK:", model.input_conv.weight.grad.abs().sum().item() > 0)