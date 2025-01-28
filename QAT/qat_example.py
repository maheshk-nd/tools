import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization as quantization
from torch.utils.data import DataLoader, TensorDataset

# Step 1: Generate Random Dataset
def generate_dataset(m, b, num_samples):
    x = torch.rand(num_samples, 1) * 10  # Random x values in range [0, 10]
    y = m * x + b #+ torch.randn(num_samples, 1)  # Adding some noise
    return x, y

# Step 2: Define a Simple Model with QuantStub and DeQuantStub
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.quant = quantization.QuantStub()  # Quantization layer
        self.fc = nn.Linear(1, 1)             # Linear layer
        self.dequant = quantization.DeQuantStub()  # Dequantization layer

    def forward(self, x):
        x = self.quant(x)    # Quantize the input
        x = self.fc(x)       # Forward pass through the linear layer
        x = self.dequant(x)  # Dequantize the output
        return x

# Step 3: Quantization Aware Training Pipeline
def train_model(m, b, num_samples, epochs=200, learning_rate=0.001):
    # Generate dataset
    x, y = generate_dataset(m, b, num_samples)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize model
    model = SimpleModel()
    model.train()

    # Prepare for quantization aware training
    model.qconfig = quantization.get_default_qat_qconfig('fbgemm')
    quantization.prepare_qat(model, inplace=True)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Convert to quantized model
    quantization.convert(model, inplace=True)

    # Save the quantized model
    torch.save(model.state_dict(), "quantized_model.pth")

    return model

# Step 4: Test the Quantized Model
def test_model(model, x_test):
    model.eval()
    # model.to('cpu')
    # x_test = x_test.to('cpu')

    with torch.no_grad():
        predictions = model(x_test)
    return predictions

if __name__ == "__main__":
    # User-defined parameters
    m = 2.5  # Slope
    b = 1.0  # Intercept
    num_samples = 10000

    # Train the model with QAT
    quantized_model = train_model(m, b, num_samples)

    # Test the quantized model
    torch.backends.quantized.engine = 'qnnpack'  # Enable quantized backend
    x_test = torch.tensor([[1.0], [2.0], [3.0], [4.0]])  # Example test inputs
    predictions = test_model(quantized_model, x_test)

    print("Test Inputs:", x_test.squeeze().tolist())
    print("Predicted Outputs:", predictions.squeeze().tolist())