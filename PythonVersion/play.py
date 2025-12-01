from libs.Training import *
from stockfish import Stockfish
import chess
import chess.svg
import webbrowser
import os

def display_board (board):
    # Generate SVG representation of the board
    svg_board = chess.svg.board(board=board)
    
    # Define file path
    file_path = os.path.abspath("current_board.html")
    
    # Write SVG to an HTML file
    with open(file_path, "w") as f:
        f.write(svg_board)
    
    # Open the HTML file in the default web browser
    webbrowser.open(f"file://{file_path}")

def play(Elo):
    # load best model
    saved_model = Model()
    
    # Update with your specific path logic
    f = open("Z:/Coding/GitHub/Python/ChessEngine/data/savedModels/bestModel2.txt", "r")
    bestLoss = float(f.readline())
    model_path = f.readline().strip() # Added strip() to remove newline characters
    f.close()

    saved_model.load_state_dict(torch.load(model_path))
    
    # Setup Stockfish
    stockfish = Stockfish(path=r"Z:\Chess\stockfish\stockfish-windows-x86-64-avx2.exe")
    stockfish.reset_engine_parameters()
    stockfish.set_elo_rating(Elo)
    stockfish.set_skill_level(0) # Be careful: Skill level 0 is very weak, might override Elo setting depending on version
    
    board = chess.Board()
    allMoves = [] 

    MAX_NUMBER_OF_MOVES = 150
    current_move_number = 0
    
    # Use a loop that respects game over, not just a range
    for i in tqdm(range(MAX_NUMBER_OF_MOVES)): 
        current_move_number += 1
        # AI Move (White)
        if board.is_game_over():
            break

        try:
            move = saved_model.predict(board)
            board.push(move)
            allMoves.append(str(move))
        except Exception as e:
            print(f"Error during AI move: {e}")
            break

        # Check if AI won immediately after moving
        if board.is_game_over():
            break

        # Stockfish Move (Black)
        stockfish.set_position(allMoves)
        stockfishMove = stockfish.get_best_move_time(500) # Give it 0.5 second (500ms) to think
        
        if stockfishMove is None:
            break
            
        allMoves.append(stockfishMove)
        stockfishMove = chess.Move.from_uci(stockfishMove)
        board.push(stockfishMove)

        print(board) #Display board state in console
        print("*"*10)

        # Check if Stockfish won immediately after moving
        if board.is_game_over():
            break

    # --- Game Over Analysis ---
    
    if board.is_game_over():
        print("Game Over!")
        if board.is_checkmate():
            if board.turn == chess.WHITE:
                print("Stockfish (Black) Won!")
            else:
                print("You (White) Won!")
        elif board.is_stalemate():
            print("Draw by Stalemate")
        elif board.is_insufficient_material():
            print("Draw by Insufficient Material")
        print("Total Moves Played:", current_move_number)
        
    return board

if __name__ == "__main__":
    user_input = input("Please enter an integer of elo you want to play against: ")
    try:
        ChessBoard = play(user_input)
        display_board(ChessBoard)
    except Exception as e:
        print("An error occurred:", str(e))