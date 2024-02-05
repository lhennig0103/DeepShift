import torch
import time

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available. Running a simple operation on GPU.")

    # Set the device to GPU
    device = torch.device("cuda")

    # Create tensors
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    y = torch.tensor([4.0, 5.0, 6.0], device=device)
    time.sleep(10)
    # Perform a simple operation (e.g., vector addition)
    z = x + y

    print(f"Result of the operation: {z}")
    print(f"Device on which the operation was performed: {z.device}")
else:
    print("CUDA is not available.")
