import torch
import sys

# Import your network
from hebbian_locomotion.networks.hebbian_neural_net import HebbianNet

def main():
    print("--- CUDA STATUS ---")
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not detected by PyTorch!")
        sys.exit(1)
        
    num_gpus = torch.cuda.device_count()
    print(f"GPUs Detected: {num_gpus}")
    print(f"Current Device Index: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    print("\n--- NETWORK TEST ---")
    try:
        # 1. Initialize Network (Popsize=128 to simulate parallel agents)
        # The class internal .cuda() calls will place weights on the default GPU (cuda:0)
        net = HebbianNet(popsize=128, sizes=[27, 64, 32, 18], norm_mode='max')
        
        # 2. Verify Weight Location
        # We check the device of the first weight matrix
        weight_device = net.weights[0].device
        print(f"Network Weights initialized on: {weight_device}")
        
        # 3. Create Dummy Input
        # CRITICAL: Since the net is on .cuda(), inputs MUST also be on .cuda()
        dummy_input = torch.randn(128, 27).cuda()
        print(f"Input Tensor created on: {dummy_input.device}")
        
        # 4. Run Forward Pass
        output = net.forward(dummy_input)
        
        print(f"Forward pass successful. Output shape: {output.shape}")
        print(f"Output device: {output.device}")
        
        if "cuda" in str(output.device):
            print("\nSUCCESS: The Hebbian Network is running on the GPU.")
        else:
            print("\nFAILURE: Output is on CPU.")
            
    except Exception as e:
        print(f"\nCRASHED: {e}")
        # Common error is device mismatch (Input on CPU, Net on GPU)

if __name__ == "__main__":
    main()