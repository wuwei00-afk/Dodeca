import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- THE IBM LIBRARIES (THE SOUL) ---
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN

# --- CONFIGURATION ---
INPUT_DIM = 8       # 8-Channel Voxel (RGB, IR, UV, Normals)
HIDDEN_DIM = 16
N_QUBITS = 4        # The Quantum Core Size

class QuantumLayer(nn.Module):
    """
    THE SCHRÖDINGER VALVE.
    This layer doesn't use math; it uses Quantum Circuits.
    """
    def __init__(self):
        super().__init__()
        
        # 1. DEFINE THE CIRCUIT
        qc = QuantumCircuit(N_QUBITS)
        
        # Parameters (The "Weights" of the quantum brain)
        self.inputs = [Parameter(f'x{i}') for i in range(N_QUBITS)]
        self.weights = [Parameter(f'w{i}') for i in range(N_QUBITS)]
        
        # 2. ENCODE DATA (Angle Embedding)
        for i in range(N_QUBITS):
            qc.rx(self.inputs[i], i) # Rotate X based on input data
            
        # 3. ENTANGLEMENT (The "Connection")
        # This links the qubits together so they act as one
        for i in range(N_QUBITS-1):
            qc.cx(i, i+1)
        qc.cx(N_QUBITS-1, 0) # Close the ring
        
        # 4. PROCESSING (Variational Rotation)
        for i in range(N_QUBITS):
            qc.ry(self.weights[i], i) # Rotate Y based on learning
            
        # 5. CREATE THE PYTORCH BRIDGE
        # This allows the Quantum Circuit to talk to the Neural Network
        qnn = EstimatorQNN(
            circuit=qc,
            input_params=self.inputs,
            weight_params=self.weights
        )
        self.q_layer = TorchConnector(qnn)

    def forward(self, x):
        return self.q_layer(x)

class TitanHybrid(nn.Module):
    def __init__(self):
        super().__init__()
        print("[INIT] Building TITAN HYBRID (PyTorch + Qiskit)...")
        
        # 1. SENSORY COMPRESSION (Classical)
        # Squeeze 8 channels down to 4 for the Quantum Core
        self.pre_process = nn.Linear(INPUT_DIM, N_QUBITS)
        
        # 2. THE QUANTUM CORE (IBM Qiskit)
        # Thinking happens in Superposition here
        self.quantum_brain = QuantumLayer()
        
        # 3. MOTOR OUTPUT (Classical)
        # Expand thought back into Action (4 Motors)
        self.post_process = nn.Linear(N_QUBITS, 4)

    def forward(self, x):
        # A. Compressing Data...
        x = torch.tanh(self.pre_process(x))
        
        # B. Entering Quantum State...
        x = self.quantum_brain(x) # <--- THIS IS THE QUANTUM STEP
        
        # C. Translating to Action...
        x = torch.sigmoid(self.post_process(x))
        return x

# --- EXECUTION ---
if __name__ == "__main__":
    # Initialize
    model = TitanHybrid()
    
    # Simulate a Data Packet (Batch of 1, 8 Channels)
    # [Vis_R, Vis_G, Vis_B, IR, UV, Gyro_X, Gyro_Y, Gyro_Z]
    data = torch.randn(1, INPUT_DIM)
    
    print(f"\n[INPUT] Feeding Sensor Data: {data.detach().numpy()}")
    print("[PROCESS] Mapping to Qubits -> Entangling -> Collapsing...")
    
    # Forward Pass
    output = model(data)
    
    print(f"[OUTPUT] Motor Controls: {output.detach().numpy()}")
    print("\n[STATUS] SUCCESS. The Logic passed through the Quantum Circuit.")