#MS Begin

#Multi processing game mining
import chess, time, uuid, multiprocessing, random
from stockfish import Stockfish
import numpy as np
import datetime
import psutil

# Helper functions
def checkEndCondition(board):
  return (
    board.is_checkmate() or
    board.is_stalemate() or
    board.is_insufficient_material() or
    board.can_claim_threefold_repetition() or
    board.can_claim_fifty_moves() or
    board.can_claim_draw()
  )

def saveData(moves, positions):
  moves = np.array(moves).reshape(-1, 1)
  positions = np.array(positions).reshape(-1,1)
  movesAndPositions = np.concatenate((moves, positions), axis=1)

  nextUuid = uuid.uuid4()

  np.save(f"Z:/Coding/GitHub/Python/ChessEngine/data/rawData/movesAndPositions{nextUuid}.npy", movesAndPositions)
  #np.save(f"../data/rawData/movesAndPositions{nextUuid}.npy", movesAndPositions)
  print(f"Saved successfully as ../data/rawData/movesAndPositions{nextUuid}.npy")

# Main game-mining function
def mineGames(numGames: int, id: int):
  print(f"Process {id} starting to mine {numGames} games...")
  
  # Initialize stockfish inside the process. Each process needs its own instance.
  stockfish = Stockfish(path="Z:\Chess\stockfish\stockfish-windows-x86-64-avx2.exe")
  MAX_MOVES = 500

  for i in range(numGames):
    board = chess.Board()
    currentGameMoves = []
    currentGamePositions = []
    stockfish.set_position([])

    for _ in range(MAX_MOVES):
      moves = stockfish.get_top_moves(3)
      if len(moves) == 0:
        break
      elif len(moves) == 1:
        move = moves[0]["Move"]
      elif len(moves) == 2:
        move = random.choices(moves, weights=(80, 20), k=1)[0]["Move"]
      else:
        move = random.choices(moves, weights=(80, 15, 5), k=1)[0]["Move"]

      currentGamePositions.append(stockfish.get_fen_position())
      currentGameMoves.append(move)

      move_obj = chess.Move.from_uci(str(move))
      board.push(move_obj)
      stockfish.set_position(currentGameMoves)

      if checkEndCondition(board):
        break

    saveData(currentGameMoves, currentGamePositions)
  
  print(f"Process {id} finished mining.")

# Multiprocessing launcher
if __name__ == "__main__":
  core_count = psutil.cpu_count(logical=False)
  print(f"Your cpu has {core_count} cores. When choosing number of cores to use, enter a number 1-2 cores lower than total amount, so you won't slow your pc down.")

  num_processes = int(input("Enter total number of cores you would like to utilize: "))
  total_games = int(input("Enter total number of games you want to mine: "))
  

  start_time = time.time()
  
  games_per_process = total_games // num_processes

  processes = []
  for i in range(num_processes):
    p = multiprocessing.Process(target=mineGames, args=(games_per_process, i))
    p.start()
    processes.append(p)

  for p in processes:
    p.join()

  end_time = time.time()
  print(f"Total time taken: {end_time - start_time} seconds")
  elapsed = str(datetime.timedelta(seconds=end_time - start_time))
  print(f"Elapsed time: {elapsed}")

  print("All games mined successfully!")
#MS End