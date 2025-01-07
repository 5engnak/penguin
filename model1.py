import torch
import torch.nn as nn

# Define the PyTorch model
class PenguinModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(PenguinModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fc(x)


# Save a pre-trained PyTorch model for use
def save_pytorch_model():
    input_size = 7  # Number of features
    output_size = 3  # Number of classes
    model = PenguinModel(input_size, output_size)
    
    # Simulate training (dummy weights for demonstration)
    model.load_state_dict({
        "fc.0.weight": torch.randn(16, input_size),
        "fc.0.bias": torch.randn(16),
        "fc.2.weight": torch.randn(8, 16),
        "fc.2.bias": torch.randn(8),
        "fc.4.weight": torch.randn(output_size, 8),
        "fc.4.bias": torch.randn(output_size),
    })
    torch.save(model.state_dict(), "penguin_pytorch_model.pth")


save_pytorch_model()
