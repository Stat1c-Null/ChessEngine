import numpy as np
import torch
from libs.CNNHelperFunctions import *
import chess

encode = input("encode all moves and positions in CNN format? yes/no")
if encode == "yes" or encode == "y":
  encodeAllMovesAndPositions()

# Test encoding
board = chess.Board()
encoded = encodeBoard(board)

# Load one file
positions = np.load('Z:/Coding/GitHub/Python/ChessEngine/data/preparedData/positions21c687d6-df39-4ef4-90e6-a0e964d45272.npy')
print(f"Shape: {positions.shape}")  # Should be (N, 14, 8, 8)

# Check one position
print(f"First position shape: {positions[0].shape}")  # Should be (14, 8, 8)

print(f"Encoded board shape: {encoded.shape}")  # Should be (14, 8, 8)

# Test with CNN
tensor = torch.from_numpy(encoded).unsqueeze(0)  # Add batch dim
print(f"Tensor shape: {tensor.shape}")  # Should be (1, 14, 8, 8)

# This should work now
from libs.CNNTraining import ChessCNN
model = ChessCNN()
output = model(tensor)
print(f"Output shape: {output.shape}")  # Should be (1, 4672)
print("Everything works!")