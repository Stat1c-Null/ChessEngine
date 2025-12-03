import numpy as np
import chess
import glob
from gym_chess.alphazero.move_encoding import utils
from typing import Optional
import uuid
import os


# ============= PIECE VALUES =============
# Standard piece values used in chess
PIECE_VALUES = {
  chess.PAWN: 1.0,
  chess.KNIGHT: 3.0,
  chess.BISHOP: 3.0,
  chess.ROOK: 5.0,
  chess.QUEEN: 9.0,
  chess.KING: 0.0  # King has no material value (game ends if lost)
}


#helper functions:
def checkEndCondition(board):
  if (board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material() or board.can_claim_threefold_repetition() or board.can_claim_fifty_moves() or board.can_claim_draw()):
    return True
  return False


def saveData(moves, positions):
  moves = np.array(moves).reshape(-1, 1)
  positions = np.array(positions).reshape(-1,1)
  movesAndPositions = np.concatenate((moves, positions), axis = 1)

  nextUuid = uuid.uuid4()
  np.save(f"../../data/rawData/movesAndPositions{nextUuid}.npy", movesAndPositions)
  print(f"Saved successfully as ../data/rawData/movesAndPositions{nextUuid}.npy")


def runGame(numMoves, index = 0):
  """run a game you stored"""
  raw_data_dir = "../../data/rawData"
  filesByLastmod = sorted(filter(os.path.isfile, glob.glob(raw_data_dir + '/*.npy')), key = os.path.getmtime)
  filename = filesByLastmod[index]
  testing = np.load(filename)
  moves = testing[:, 0]
  if (numMoves > len(moves)):
    print("Must enter a lower number of moves than maximum game length. Game length here is: ", len(moves))
    return

  testBoard = chess.Board()

  for i in range(numMoves):
    move = moves[i]
    testBoard.push_san(move)

  print(filename)
  return testBoard


#fixing encoding funcs from openai

def encodeKnight(move: chess.Move):
  _NUM_TYPES: int = 8

  #: Starting point of knight moves in last dimension of 8 x 8 x 73 action array.
  _TYPE_OFFSET: int = 56

  #: Set of possible directions for a knight move, encoded as 
  #: (delta rank, delta square).
  _DIRECTIONS = utils.IndexedTuple(
    (+2, +1),
    (+1, +2),
    (-1, +2),
    (-2, +1),
    (-2, -1),
    (-1, -2),
    (+1, -2),
    (+2, -1),
  )

  from_rank, from_file, to_rank, to_file = utils.unpack(move)

  delta = (to_rank - from_rank, to_file - from_file)
  is_knight_move = delta in _DIRECTIONS
  
  if not is_knight_move:
    return None

  knight_move_type = _DIRECTIONS.index(delta)
  move_type = _TYPE_OFFSET + knight_move_type

  action = np.ravel_multi_index(
    multi_index=((from_rank, from_file, move_type)),
    dims=(8, 8, 73)
  )

  return action

def encodeQueen(move: chess.Move):
  _NUM_TYPES: int = 56 # = 8 directions * 7 squares max. distance
  _DIRECTIONS = utils.IndexedTuple(
    (+1,  0),
    (+1, +1),
    ( 0, +1),
    (-1, +1),
    (-1,  0),
    (-1, -1),
    ( 0, -1),
    (+1, -1),
  )

  from_rank, from_file, to_rank, to_file = utils.unpack(move)

  delta = (to_rank - from_rank, to_file - from_file)

  is_horizontal = delta[0] == 0
  is_vertical = delta[1] == 0
  is_diagonal = abs(delta[0]) == abs(delta[1])
  is_queen_move_promotion = move.promotion in (chess.QUEEN, None)

  is_queen_move = (
    (is_horizontal or is_vertical or is_diagonal) 
      and is_queen_move_promotion
  )

  if not is_queen_move:
    return None

  direction = tuple(np.sign(delta))
  distance = np.max(np.abs(delta))

  direction_idx = _DIRECTIONS.index(direction)
  distance_idx = distance - 1

  move_type = np.ravel_multi_index(
    multi_index=([direction_idx, distance_idx]),
    dims=(8,7)
  )

  action = np.ravel_multi_index(
    multi_index=((from_rank, from_file, move_type)),
    dims=(8, 8, 73)
  )

  return action

def encodeUnder(move):
  _NUM_TYPES: int = 9 # = 3 directions * 3 piece types (see below)
  _TYPE_OFFSET: int = 64
  _DIRECTIONS = utils.IndexedTuple(
    -1,
    0,
    +1,
  )
  _PROMOTIONS = utils.IndexedTuple(
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
  )

  from_rank, from_file, to_rank, to_file = utils.unpack(move)

  is_underpromotion = (
    move.promotion in _PROMOTIONS 
    and from_rank == 6 
    and to_rank == 7
  )

  if not is_underpromotion:
    return None

  delta_file = to_file - from_file

  direction_idx = _DIRECTIONS.index(delta_file)
  promotion_idx = _PROMOTIONS.index(move.promotion)

  underpromotion_type = np.ravel_multi_index(
    multi_index=([direction_idx, promotion_idx]),
    dims=(3,3)
  )

  move_type = _TYPE_OFFSET + underpromotion_type

  action = np.ravel_multi_index(
    multi_index=((from_rank, from_file, move_type)),
    dims=(8, 8, 73)
  )

  return action

def encodeMove(move: str, board) -> int:
  move = chess.Move.from_uci(move)
  if board.turn == chess.BLACK:
    move = utils.rotate(move)

  action = encodeQueen(move)

  if action is None:
    action = encodeKnight(move)

  if action is None:
    action = encodeUnder(move)

  if action is None:
    raise ValueError(f"{move} is not a valid move")

  return action


# ============= FIXED ENCODEBOARD FUNCTION =============
# This is the KEY fix - returns shape (15, 8, 8) in correct format
def encodeBoard(board: chess.Board):
  """
  Encode chess board for CNN input
  Returns: numpy array of shape (15, 8, 8)
  
  Channels (dimension 0):
  0-5: White pieces (Pawn, Knight, Bishop, Rook, Queen, King)
  6-11: Black pieces (Pawn, Knight, Bishop, Rook, Queen, King)
  12: En passant target square
  13: Castling rights
  14: Piece values (normalized material value at each square)
  
  CRITICAL: Returns (channels, height, width) for PyTorch Conv2D
  """
  # Initialize 15 channels of 8×8 boards
  # Shape: (15, 8, 8) - channels first!
  encoded = np.zeros((15, 8, 8), dtype=np.float32)
  
  piece_to_channel = {
    chess.PAWN: 0, 
    chess.KNIGHT: 1, 
    chess.BISHOP: 2,
    chess.ROOK: 3, 
    chess.QUEEN: 4, 
    chess.KING: 5
  }
  
  # Encode pieces
  for square in chess.SQUARES:
    piece = board.piece_at(square)
    if piece:
      rank = chess.square_rank(square)  # 0-7 (row)
      file = chess.square_file(square)  # 0-7 (column)
      
      # Get base channel for piece type
      channel = piece_to_channel[piece.piece_type]
      
      # Black pieces go in channels 6-11
      if piece.color == chess.BLACK:
        channel += 6
      
      # Set the bit: encoded[channel, rank, file] = 1
      encoded[channel, rank, file] = 1.0
  
  # Encode en passant
  if board.ep_square:
    rank = chess.square_rank(board.ep_square)
    file = chess.square_file(board.ep_square)
    encoded[12, rank, file] = 1.0
  
  # Encode castling rights (full board gets the value)
  castling_value = 0.0
  if board.has_kingside_castling_rights(chess.WHITE):
    castling_value += 0.25
  if board.has_queenside_castling_rights(chess.WHITE):
    castling_value += 0.25
  if board.has_kingside_castling_rights(chess.BLACK):
    castling_value += 0.25
  if board.has_queenside_castling_rights(chess.BLACK):
    castling_value += 0.25
  encoded[13, :, :] = castling_value
  
  # Encode piece values (channel 14)
  # Positive for white pieces, negative for black pieces
  # Normalized by dividing by max piece value (9.0 for queen)
  for square in chess.SQUARES:
    piece = board.piece_at(square)
    if piece:
      rank = chess.square_rank(square)
      file = chess.square_file(square)
      value = PIECE_VALUES[piece.piece_type]
      
      # Normalize to [-1, 1] range
      # White pieces: positive values, Black pieces: negative values
      if piece.color == chess.WHITE:
        encoded[14, rank, file] = value / 9.0  # Normalize by queen value
      else:
        encoded[14, rank, file] = -value / 9.0  # Negative for black
  
  return encoded


def encodeBoardFromFen(fen: str) -> np.array:
  """Encode a chess position from FEN string"""
  board = chess.Board(fen)
  return encodeBoard(board)


# Saves data in the correct shape for CNN training
def encodeAllMovesAndPositions():
  """
  Encodes all moves and positions from raw data files and saves them as prepared data files.
  
  IMPORTANT: Saves boards as (14, 8, 8) for CNN compatibility
  
  This function reads raw data files containing moves and positions, encodes each move and position,
  and saves the encoded data as prepared data files. It uses the `encodeMove` and `encodeBoardFromFen`
  functions to perform the encoding.
  
  Returns:
    None
  """
  board = chess.Board() #this is used to change whose turn it is so that the encoding works
  board.turn = False #set turn to black first, changed on first run

  #find all files in folder:
  files = (glob.glob(r"Z:/Coding/GitHub/Python/ChessEngine/data/rawData/movesAndPositions*.npy"))
  
  print(f"Found {len(files)} files to encode")
  
  for idx, f in enumerate(files):
    print(f"Processing file {idx+1}/{len(files)}: {f}")
    
    movesAndPositions = np.load(f'{f}', allow_pickle=True)
    moves = movesAndPositions[:,0]
    positions = movesAndPositions[:,1]
    encodedMoves = []
    encodedPositions = []

    for i in range(len(moves)):
      board.turn = (not board.turn) #swap turns
      try:
        encodedMoves.append(encodeMove(moves[i], board)) 
        encodedPositions.append(encodeBoardFromFen(positions[i]))
      except:
        try:
          board.turn = (not board.turn) #change turn, since you skip moves sometimes, you might need to change turn
          encodedMoves.append(encodeMove(moves[i], board)) 
          encodedPositions.append(encodeBoardFromFen(positions[i]))
        except Exception as e:
          print(f'Error in file: {f}')
          print(f"Turn: {board.turn}")
          print(f"Move: {moves[i]}")
          print(f"Position: {positions[i]}")
          print(f"Index: {i}")
          print(f"Error: {e}")
          break

    # Convert to numpy arrays
    encodedMoves = np.array(encodedMoves)
    encodedPositions = np.array(encodedPositions)
    
    # Verify shapes
    print(f"  Encoded {len(encodedMoves)} moves")
    print(f"  Positions shape: {encodedPositions.shape}")  # Should be (N, 15, 8, 8)
    
    # Save with same UUID
    currUuid = f.split("movesAndPositions")[-1].split(".npy")[0]
    np.save(f'Z:/Coding/GitHub/Python/ChessEngine/data/preparedData/moves{currUuid}', encodedMoves)
    np.save(f'Z:/Coding/GitHub/Python/ChessEngine/data/preparedData/positions{currUuid}', encodedPositions)
    
    print(f"  Saved as moves{currUuid}.npy and positions{currUuid}.npy")

  print("\nAll files encoded successfully")


#helper methods:

#decoding moves from idx to uci notation
def _decodeKnight(action: int) -> Optional[chess.Move]:
  """
  Decodes the given action into a knight move in the chess game.

  Args:
    action (int): The action to decode.

  Returns:
    Optional[chess.Move]: The decoded knight move as a chess.Move object, or None if the action is not a valid knight move.
  """

  _NUM_TYPES: int = 8

  #: Starting point of knight moves in last dimension of 8 x 8 x 73 action array.
  _TYPE_OFFSET: int = 56

  #: Set of possible directions for a knight move, encoded as 
  #: (delta rank, delta square).
  _DIRECTIONS = utils.IndexedTuple(
    (+2, +1),
    (+1, +2),
    (-1, +2),
    (-2, +1),
    (-2, -1),
    (-1, -2),
    (+1, -2),
    (+2, -1),
  )

  from_rank, from_file, move_type = np.unravel_index(action, (8, 8, 73))

  is_knight_move = (
    _TYPE_OFFSET <= move_type
    and move_type < _TYPE_OFFSET + _NUM_TYPES
  )

  if not is_knight_move:
    return None

  knight_move_type = move_type - _TYPE_OFFSET

  delta_rank, delta_file = _DIRECTIONS[knight_move_type]

  to_rank = from_rank + delta_rank
  to_file = from_file + delta_file

  move = utils.pack(from_rank, from_file, to_rank, to_file)
  return move

def _decodeQueen(action: int) -> Optional[chess.Move]:

  _NUM_TYPES: int = 56 # = 8 directions * 7 squares max. distance

  #: Set of possible directions for a queen move, encoded as 
  #: (delta rank, delta square).
  _DIRECTIONS = utils.IndexedTuple(
    (+1,  0),
    (+1, +1),
    ( 0, +1),
    (-1, +1),
    (-1,  0),
    (-1, -1),
    ( 0, -1),
    (+1, -1),
  )
  from_rank, from_file, move_type = np.unravel_index(action, (8, 8, 73))
  
  is_queen_move = move_type < _NUM_TYPES

  if not is_queen_move:
    return None

  direction_idx, distance_idx = np.unravel_index(
    indices=move_type,
    shape=(8,7)
  )

  direction = _DIRECTIONS[direction_idx]
  distance = distance_idx + 1

  delta_rank = direction[0] * distance
  delta_file = direction[1] * distance

  to_rank = from_rank + delta_rank
  to_file = from_file + delta_file

  move = utils.pack(from_rank, from_file, to_rank, to_file)
  return move

def _decodeUnderPromotion(action):
  _NUM_TYPES: int = 9 # = 3 directions * 3 piece types (see below)

  #: Starting point of underpromotions in last dimension of 8 x 8 x 73 action 
  #: array.
  _TYPE_OFFSET: int = 64

  #: Set of possibel directions for an underpromotion, encoded as file delta.
  _DIRECTIONS = utils.IndexedTuple(
    -1,
    0,
    +1,
  )

  #: Set of possibel piece types for an underpromotion (promoting to a queen
  #: is implicitly encoded by the corresponding queen move).
  _PROMOTIONS = utils.IndexedTuple(
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
  )

  from_rank, from_file, move_type = np.unravel_index(action, (8, 8, 73))

  is_underpromotion = (
    _TYPE_OFFSET <= move_type
    and move_type < _TYPE_OFFSET + _NUM_TYPES
  )

  if not is_underpromotion:
    return None

  underpromotion_type = move_type - _TYPE_OFFSET

  direction_idx, promotion_idx = np.unravel_index(
    indices=underpromotion_type,
    shape=(3,3)
  )

  direction = _DIRECTIONS[direction_idx]
  promotion = _PROMOTIONS[promotion_idx]

  to_rank = from_rank + 1
  to_file = from_file + direction

  move = utils.pack(from_rank, from_file, to_rank, to_file)
  move.promotion = promotion

  return move

#primary decoding function, the ones above are just helper functions
def decodeMove(action: int, board) -> chess.Move:
  move = _decodeQueen(action)
  is_queen_move = move is not None

  if not move:
    move = _decodeKnight(action)

  if not move:
    move = _decodeUnderPromotion(action)

  if not move:
    raise ValueError(f"{action} is not a valid action")

  # Actions encode moves from the perspective of the current player. If
  # this is the black player, the move must be reoriented.
  turn = board.turn
  
  if turn == False: #black to move
    move = utils.rotate(move)

  # Moving a pawn to the opponent's home rank with a queen move
  # is automatically assumed to be queen underpromotion. However,
  # since queenmoves has no reference to the board and can thus not
  # determine whether the moved piece is a pawn, we have to add this
  # information manually here
  if is_queen_move:
    to_rank = chess.square_rank(move.to_square)
    is_promoting_move = (
      (to_rank == 7 and turn == True) or 
      (to_rank == 0 and turn == False)
    )

    piece = board.piece_at(move.from_square)
    if piece is None: #NOTE I added this, not entirely sure if it's correct
      return None
    is_pawn = piece.piece_type == chess.PAWN

    if is_pawn and is_promoting_move:
      move.promotion = chess.QUEEN

  return move