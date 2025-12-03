from libs.CNNTraining import *
from stockfish import Stockfish
import chess
import chess.svg
import webbrowser
import os, time
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setGeometry(100, 100, 1100, 1100)

        self.widgetSvg = QSvgWidget(parent=self)
        self.widgetSvg.setGeometry(10, 10, 1080, 1080)

        self.chessboard = chess.Board()

        self.chessboardSvg = chess.svg.board(self.chessboard).encode("UTF-8")
        self.widgetSvg.load(self.chessboardSvg)
    
    def draw_board(self, event):
         self.chessboardSvg = chess.svg.board(self.chessboard).encode("UTF-8")
         self.widgetSvg.load(self.chessboardSvg) 

def play(Elo, main_window=None):
    # load best model
    saved_model = ChessCNN()

    # Update with your specific path logic
    f = open("Z:/Coding/GitHub/Python/ChessEngine/data/savedModels/bestModel.txt", "r")
    bestLoss = float(f.readline())
    model_path = f.readline().strip()  # Added strip() to remove newline characters
    f.close()

    saved_model.load_state_dict(torch.load(model_path))

    # Setup Stockfish
    stockfish = Stockfish(path=r"Z:\Chess\stockfish\stockfish-windows-x86-64-avx2.exe")
    stockfish.reset_engine_parameters()
    stockfish.set_elo_rating(Elo)
    stockfish.set_skill_level(0)  # Be careful: Skill level 0 is very weak, might override Elo setting depending on version

    board = chess.Board()
    allMoves = []

    MAX_NUMBER_OF_MOVES = 150
    current_move_number = 0

    # Use a loop that respects game over, not just a range
    for i in range(MAX_NUMBER_OF_MOVES):
        current_move_number += 1

        # AI Move (White)
        if board.is_game_over():
            break

        try:
            move = saved_model.predict(board)
            board.push(move)
            allMoves.append(str(move))

            # Update the MainWindow dynamically
            if main_window:
                main_window.chessboard = board
                main_window.draw_board(None)
                QApplication.processEvents()

        except Exception as e:
            print(f"Error during AI move: {e}")
            break

        # Check if AI won immediately after moving
        if board.is_game_over():
            break

        # Stockfish Move (Black)
        stockfish.set_position(allMoves)
        stockfishMove = stockfish.get_best_move_time(500)  # Give it 0.1 second (100ms) to think

        if stockfishMove is None:
            break

        allMoves.append(stockfishMove)
        stockfishMove = chess.Move.from_uci(stockfishMove)
        board.push(stockfishMove)

        # Update the MainWindow dynamically
        if main_window:
            main_window.chessboard = board
            main_window.draw_board(None)
            QApplication.processEvents()

        # Check if Stockfish won immediately after moving
        if board.is_game_over():
            break
        
        time.sleep(0.5)
    # Game Over Analysis
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
        app = QApplication([])
        main_window = MainWindow()
        main_window.show()

        ChessBoard = play(user_input, main_window=main_window)

        app.exec_()
    except Exception as e:
        print("An error occurred:", str(e))