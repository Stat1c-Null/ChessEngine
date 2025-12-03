import glob
from datetime import datetime
from pathlib import Path

import chess
import numpy as np
import torch
from libs.CNNHelperFunctions import *
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import torch
import torch.nn as nn
import chess
import numpy as np


def LoadingTrainingData(FRACTION_OF_DATA=1, BATCH_SIZE=64):
  allMoves = []
  allBoards = []

  files = (
      glob.glob(r"Z:/Coding/GitHub/Python/ChessEngine/data/preparedData/moves*.npy"))

  for f in files:
    currUuid = f.split("moves")[-1].split(".npy")[0]
    try:
      moves = np.load(
          f"Z:/Coding/GitHub/Python/ChessEngine/data/preparedData/moves{currUuid}.npy", allow_pickle=True)
      boards = np.load(
          f"Z:/Coding/GitHub/Python/ChessEngine/data/preparedData/positions{currUuid}.npy", allow_pickle=True)
      if (len(moves) != len(boards)):
        print("ERROR ON i = ", currUuid, len(moves), len(boards))
      allMoves.extend(moves)
      allBoards.extend(boards)
    except:
      print("error: could not load ", currUuid, ", but is still going")
      pass

  allMoves = np.array(allMoves)[:(int(len(allMoves) * FRACTION_OF_DATA))]
  allBoards = np.array(allBoards)[:(int(len(allBoards) * FRACTION_OF_DATA))]
  assert len(allMoves) == len(allBoards), "MUST BE OF SAME LENGTH"

  trainDataIdx = int(len(allMoves) * 0.8)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  allBoards = torch.from_numpy(np.asarray(allBoards)).to(device)
  allMoves = torch.from_numpy(np.asarray(allMoves)).to(device)

  training_set = torch.utils.data.TensorDataset(
      allBoards[:trainDataIdx], allMoves[:trainDataIdx])
  test_set = torch.utils.data.TensorDataset(
      allBoards[trainDataIdx:], allMoves[trainDataIdx:])
  
  # Create data loaders for our datasets; shuffle for training, not for validation
  training_loader = torch.utils.data.DataLoader(
      training_set, batch_size=BATCH_SIZE, shuffle=True)
  validation_loader = torch.utils.data.DataLoader(
      test_set, batch_size=BATCH_SIZE, shuffle=False)

  print(f"loaded {len(allMoves)} moves and positions")
  return [training_loader, validation_loader]

# model
class ChessCNN(torch.nn.Module):
  
  def __init__(self):
    super(ChessCNN, self).__init__()
    
    # Input: (batch_size, 15, 8, 8)
    # 15 channels: 6 piece types × 2 colors + en passant + castling + piece values
    # 8×8: chess board dimensions
    
    self.OUTPUT_SIZE = 4672  # Number of possible moves
    
    # ============= CONVOLUTIONAL LAYERS =============
    # These extract spatial features from the board
    
    # Conv Layer 1: Learn basic piece patterns
    # Input: 15 channels - Output: 32 channels (includes piece value channel)
    # Kernel size 3×3: looks at 3×3 squares at a time
    # Padding=1: keeps the 8×8 size
    self.conv1 = nn.Conv2d(
      in_channels=15,    # Input channels (piece types + values)
      out_channels=32,   # Output channels (learned features)
      kernel_size=3,     # 3×3 filter (looks at piece + neighbors)
      padding=1          # Keep 8×8 size
    )
    
    # Batch Normalization: Stabilizes training by normalizing activations
    # Helps the network train faster and more reliably
    self.bn1 = nn.BatchNorm2d(32)
    
    # Conv Layer 2: Learn more complex patterns
    # Input: 32 channels - Output: 64 channels
    # Now learning combinations of the features from conv1
    self.conv2 = nn.Conv2d(
      in_channels=32,
      out_channels=64,
      kernel_size=3,
      padding=1
    )
    self.bn2 = nn.BatchNorm2d(64)
    
    # Conv Layer 3: Even deeper patterns
    # Input: 64 channels - Output: 128 channels
    # Learning high-level strategic patterns (tactics, structure)
    self.conv3 = nn.Conv2d(
      in_channels=64,
      out_channels=128,
      kernel_size=3,
      padding=1
    )
    self.bn3 = nn.BatchNorm2d(128)
    
    # Activation function: ReLU is standard for CNNs
    # More effective than Tanh for deep networks
    self.activation = nn.ReLU()
    
    # Dropout: Randomly "turns off" 30% of neurons during training
    # Prevents overfitting
    self.dropout = nn.Dropout(0.3)
    
    # ============= MATERIAL EVALUATION LAYER (Optional) =============
    # This layer specifically processes the piece value channel
    # Helps the model learn strategic material evaluation
    self.value_conv = nn.Conv2d(
      in_channels=1,     # Only the piece value channel (channel 14)
      out_channels=8,    # Learn 8 different value-based features
      kernel_size=3,
      padding=1
    )
    self.value_bn = nn.BatchNorm2d(8)
    
    # ============= FULLY CONNECTED LAYERS =============
    # After conv layers extract features, these make the final decision
    
    # After 3 conv layers with same size (due to padding=1):
    # Output is still 8×8, but with 128 channels from main path + 8 from value path
    # Flattened size: (128 + 8) × 8 × 8 = 8704
    self.fc1 = nn.Linear((128 + 8) * 8 * 8, 1024)
    self.fc2 = nn.Linear(1024, 512)
    self.fc3 = nn.Linear(512, self.OUTPUT_SIZE)
    
    # Softmax for prediction (converts to probabilities)
    self.softmax = nn.Softmax(dim=1)
  
  def forward(self, x):
    # x shape: (batch_size, 15, 8, 8)
    
    # Ensure correct data type
    x = x.to(torch.float32)
    
    # ============= SEPARATE VALUE CHANNEL PROCESSING =============
    # Extract the piece value channel (channel 14) for dedicated processing
    value_channel = x[:, 14:15, :, :]  # Shape: (batch, 1, 8, 8)
    
    # Process value channel through dedicated conv layer
    value_features = self.value_conv(value_channel)  # (batch, 8, 8, 8)
    value_features = self.value_bn(value_features)
    value_features = self.activation(value_features)
    
    # ============= MAIN CONVOLUTIONAL BLOCKS =============
    # Each block: Conv → BatchNorm → Activation → Dropout
    
    # Block 1: Basic feature extraction
    x = self.conv1(x)           # (batch, 32, 8, 8) - processes all 15 channels
    x = self.bn1(x)             # Normalize
    x = self.activation(x)      # Apply ReLU
    x = self.dropout(x)         # Prevent overfitting
    
    # Block 2: Mid-level features
    x = self.conv2(x)           # (batch, 64, 8, 8)
    x = self.bn2(x)
    x = self.activation(x)
    x = self.dropout(x)
    
    # Block 3: High-level strategic features
    x = self.conv3(x)           # (batch, 128, 8, 8)
    x = self.bn3(x)
    x = self.activation(x)
    x = self.dropout(x)
    
    # ============= CONCATENATE VALUE FEATURES =============
    # Combine main features with value-specific features
    x = torch.cat([x, value_features], dim=1)  # (batch, 136, 8, 8)
    
    # ============= FLATTEN & FULLY CONNECTED =============
    # Flatten: Convert 3D tensor to 1D for dense layers
    x = x.reshape(x.shape[0], -1)  # (batch, 8704)
    
    # Dense layers for final move prediction
    x = self.fc1(x)             # (batch, 1024)
    x = self.activation(x)
    x = self.dropout(x)
    
    x = self.fc2(x)             # (batch, 512)
    x = self.activation(x)
    x = self.dropout(x)
    
    x = self.fc3(x)             # (batch, 4672) - raw scores
    
    # Don't apply softmax here if using CrossEntropyLoss
    # CrossEntropyLoss expects raw logits (unnormalized scores)
    return x
  
  def predict(self, board: chess.Board):
    """
    Takes a chess board and returns the best legal move
    """
    with torch.no_grad():
      # Encode board - IMPORTANT: must match CNN input shape
      # Should return shape (15, 8, 8), NOT flattened
      encodedBoard = encodeBoard(board)  # Now includes piece values!
      
      # Add batch dimension: (15, 8, 8) → (1, 15, 8, 8)
      encodedBoard = np.expand_dims(encodedBoard, axis=0)
      encodedBoard = torch.from_numpy(encodedBoard)
      
      # Get model output
      res = self.forward(encodedBoard)
      probs = self.softmax(res)
      
      # Convert to numpy (remove batch dimension)
      probs = probs.cpu().numpy()[0]
      
      # Find legal move with highest probability
      while len(probs) > 0:
        moveIdx = probs.argmax()
        try:
          uciMove = decodeMove(moveIdx, board)
          if uciMove is None:
            probs = np.delete(probs, moveIdx)
            continue
          
          move = chess.Move.from_uci(str(uciMove))
          if move in board.legal_moves:
            return move
        except:
          pass
        
        probs = np.delete(probs, moveIdx)
      
      # Fallback to random legal move
      moves = list(board.legal_moves)
      if len(moves) > 0:
        print(f"Model failed, returning random move from {len(moves)} options")
        return np.random.choice(moves)
      
      print("No legal moves found")
      return None


def train_one_epoch(model, optimizer, loss_fn, epoch_index, tb_writer, training_loader):
  running_loss = 0.
  last_loss = 0.

  for i, data in enumerate(training_loader):

    inputs, labels = data

    # Zero gradients for every batch
    optimizer.zero_grad()

    # Make predictions for this batch
    outputs = model(inputs)

    # Compute the loss and its gradients
    loss = loss_fn(outputs, labels)
    loss.backward()

    # Adjust learning weights
    optimizer.step()

    # Gather data and report
    running_loss += loss.item()
    if i % 1000 == 999:
      last_loss = running_loss / 1000  # loss per batch
      # print('  batch {} loss: {}'.format(i + 1, last_loss))
      tb_x = epoch_index * len(training_loader) + i + 1
      tb_writer.add_scalar('Loss/train', last_loss, tb_x)
      running_loss = 0.

  return last_loss


def createBestModelFile():
  # first find best model if it exists:
  path = Path(
      'Z:/Coding/GitHub/Python/ChessEngine/data/savedModels/bestModel.txt')

  if not (path.is_file()):
    # create the files
    f = open(path, "w")
    # set to high number so it is overwritten with better loss
    f.write("10000000")
    f.write("\ntestPath")
    f.close()


def saveBestModel(vloss, pathToBestModel, epoch_number):
  f = open("Z:/Coding/GitHub/Python/ChessEngine/data/savedModels/bestModel.txt", "w")
  f.write(str(vloss.item()))
  f.write("\n")
  f.write(pathToBestModel)
  print("NEW BEST MODEL FOUND WITH LOSS:", vloss)


def retrieveBestModelInfo():
  f = open('Z:/Coding/GitHub/Python/ChessEngine/data/savedModels/bestModel.txt', "r")
  bestLoss = float(f.readline())
  bestModelPath = f.readline()
  f.close()
  return bestLoss, bestModelPath


# hyperparams
EPOCHS = 300
LEARNING_RATE = 0.0005
MOMENTUM = 0.9
EARLY_STOP_PATIENCE = 30  # Stop training if no improvement for 30 epochs


def runTraining():
  print("Training CNN model")

  createBestModelFile()

  bestLoss, bestModelPath = retrieveBestModelInfo()
  trainDataLoader, testDataLoader = LoadingTrainingData()

  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
  epoch_number = 0

  model = ChessCNN()
  loss_fn = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(
      model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
  
  # Learning rate scheduler - reduces LR when validation loss plateaus
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      optimizer,
      mode='min',           # Minimize validation loss
      factor=0.5,           # Reduce LR by half
      patience=15,          # Wait 15 epochs before reducing
      min_lr=1e-6           # Don't go below this LR
  )
  
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model.to(device)

  best_vloss = 1_000_000.
  epochs_without_improvement = 0  # Early stopping counter

  for epoch in tqdm(range(EPOCHS)):
    if (epoch_number % 5 == 0):
      print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(
        model, optimizer, loss_fn, epoch_number, writer, trainDataLoader)

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.

    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
      for i, vdata in enumerate(trainDataLoader):
        vinputs, vlabels = vdata
        voutputs = model(vinputs)

        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    
    # Step the learning rate scheduler based on validation loss
    scheduler.step(avg_vloss)
    
    # Get current learning rate for logging
    current_lr = optimizer.param_groups[0]['lr']

    # Enhanced progress tracking - print every 5 epochs
    if epoch_number % 5 == 0:
      improvement = "" if avg_vloss >= best_vloss else " ✓ NEW BEST"
      print('Epoch {:3d} | Train Loss: {:.6f} | Valid Loss: {:.6f} | LR: {:.2e}{}'.format(
          epoch_number + 1, avg_loss, avg_vloss, current_lr, improvement))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                       {'Training': avg_loss, 'Validation': avg_vloss},
                       epoch_number + 1)
    writer.add_scalar('Learning Rate', current_lr, epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
      best_vloss = avg_vloss
      epochs_without_improvement = 0  # Reset counter on improvement

      if (bestLoss > best_vloss):  # if better than previous best loss from all models created, save it
        model_path = 'Z:/Coding/GitHub/Python/ChessEngine/data/savedModels/model_{}_{}'.format(
            timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)
        saveBestModel(best_vloss, model_path, epoch_number)
        print(f"Saved new best model: {model_path}")
    else:
      epochs_without_improvement += 1
      if epochs_without_improvement >= EARLY_STOP_PATIENCE:
        print(f"\n\nEarly stopping triggered at epoch {epoch_number + 1}")
        print(f"No improvement for {EARLY_STOP_PATIENCE} epochs")
        print(f"Best validation loss: {best_vloss:.6f}")
        break

    epoch_number += 1

  print("\n\nBEST VALIDATION LOSS FOR ALL MODELS: ", bestLoss)
