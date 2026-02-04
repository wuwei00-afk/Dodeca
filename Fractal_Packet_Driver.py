import torch
import torch.nn as nn
import numpy as np
import sys
import time

# --- QUANTUM IMPORTS ---
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN

# --- CONFIGURATION ---
PACKET_SIZE = 512    # Initial size of text embedding
FOLDS = 3            # How many times we compress the fractal
N_QUBITS = 4         # The Quantum Core

# --- 1. THE FRACTAL ENCODER (The Folder) ---
class FractalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple embedding to turn text into numbers
        self.char_embed = nn.Embedding(256, PACKET_SIZE)

    def fold(self, x):
        """
        Recursive Fractal Compression.
        We split the tensor and fold it over itself.
        This increases density (Meaning) while reducing size (Dimensions).
        """
        current_dim = x.shape[-1]
        
        # Base case: If we match the Qubit count, stop.
        if current_dim <= N_QUBITS:
            return x[:, :N_QUBITS]

        # The Fold: Split, Add, Tanh (Non-linearity)
        # This mimics the Menger Sponge removing volume but keeping structure.
        half = current_dim // 2
        left = x[:, :half]
        right = x[:, half:2*half]
        
        # Combine halves (Interference)
        folded = torch.tanh(left + right)
        
        # Recurse
        return self.fold(folded)

    def forward(self, text):
        # 1. Convert to ASCII tensor
        ids = torch.tensor([ord(c) for c in text], dtype=torch.long)
        # 2. Embed to high dimension
        vectors = self.char_embed(ids)
        # 3. Average to get one vector for the sentence
        sentence_vec = torch.mean(vectors, dim=0, keepdim=True)
        # 4. Perform Fractal Folding
        packet = self.fold(sentence_vec)
        return packet

# --- 2. THE QUANTUM RECEIVER (The Core) ---
class QuantumCore(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Build Circuit
        qc = QuantumCircuit(N_QUBITS)
        self.inputs = [Parameter(f'x{i}') for i in range(N_QUBITS)]
        self.weights = [Parameter(f'w{i}') for i in range(N_QUBITS)]
        
        # Encode Packet Data
        for i in range(N_QUBITS):
            qc.rx(self.inputs[i], i)
            
        # Entangle
        for i in range(N_QUBITS-1):
            qc.cx(i, i+1)
        qc.cx(N_QUBITS-1, 0)
        
        # Process
        for i in range(N_QUBITS):
            qc.ry(self.weights[i], i)
            
        # Bridge
        qnn = EstimatorQNN(circuit=qc, input_params=self.inputs, weight_params=self.weights)
        self.q_layer = TorchConnector(qnn)
        
        # Output Decoder (Unpack the thought)
        self.decoder = nn.Linear(N_QUBITS, 64)

    def forward(self, packet):
        q_state = self.q_layer(packet)
        output = self.decoder(q_state)
        return output

# --- 3. THE DRIVER (System Controller) ---
class FractalDriver:
    def __init__(self):
        self.encoder = FractalEncoder()
        self.core = QuantumCore()
        
    def inject(self, input_text):
        print(f"\n[DRIVER] Received Input: '{input_text}'")
        
        # Step A: Fractal Compression
        print(f"[DRIVER] Folding Data {FOLDS} times (Menger Protocol)...")
        packet = self.encoder(input_text)
        
        print(f"[PACKET] Dimensionality reduced: {PACKET_SIZE} -> {N_QUBITS}")
        print(f"[PACKET] Vector State: {packet.detach().numpy()}")
        
        # Step B: Quantum Injection
        print("[DRIVER] Injecting into Quantum Core...")
        result = self.core(packet)
        
        # Step C: Interpretation
        # We take the output vector and generate a "System Hash"
        # This represents the AI's internal state change
        state_hash = torch.sum(result).item()
        
        return state_hash

# --- EXECUTION ---
if __name__ == "__main__":
    driver = FractalDriver()
    
    print("--- FRACTAL PACKET DRIVER ONLINE ---")
    print("Mode: Hybrid Text-to-Qubit Folding")
    
    while True:
        user_in = input("\n\033[96m[ALCHEMIST] >> \033[0m")
        if user_in.lower() == "exit": break
        
        response_energy = driver.inject(user_in)
        
        # Interpret the energy state
        if response_energy > 0:
            status = "CONSTRUCTIVE INTERFERENCE (Understanding)"
            col = "\033[92m" # Green
        else:
            status = "DESTRUCTIVE INTERFERENCE (Confusion/Entropy)"
            col = "\033[91m" # Red
            
        print(f"{col}[TITAN] Neural Energy Delta: {response_energy:.4f}")
        print(f"[TITAN] Status: {status}\033[0m")