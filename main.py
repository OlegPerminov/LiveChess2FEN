import os

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input as \
        prein_xception


from lc2fen.test_predict_board import predict_board
from lc2fen.predict_board import detect_input_board, obtain_individual_pieces
from lc2fen.predict_board import load_image

MODEL_PATH_KERAS = "/src/models/Xception.h5"
IMG_SIZE_KERAS = 299
PRE_INPUT_KERAS = prein_xception

def main_keras(model):
    """Executes Keras test board predictions."""
    print("Keras predictions")

    def obtain_pieces_probs(pieces):
        predictions = []
        for piece in pieces:
            piece_img = load_image(piece, IMG_SIZE_KERAS, PRE_INPUT_KERAS)
            predictions.append(model.predict(piece_img)[0])
        return predictions

    fen = predict_board(os.path.join("predictions", "test6.jpg"), "BL",
                        obtain_pieces_probs)

    print(fen)

__PREDS_DICT = {0: 'White Biship', 1: 'White King', 2: 'White Knight', 3: 'White Pawn', 4: 'White Queen', 5: 'White Roque ', 6: 'Space',
                7: 'Black Biship', 8: 'Black King', 9: 'Black Knight', 10: 'Black Pawn', 11: 'Black Queen', 12: 'Black Roque'}

BOARD_SIZE = 8
LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

if __name__ == "__main__":
    model = load_model(MODEL_PATH_KERAS)
    main_keras(model)
    board_path = os.path.join("predictions", "test8.jpg")
    detect_input_board(board_path)
    pieces = obtain_individual_pieces(board_path)
    print(pieces[0])
    index = 0
    output_first = ''
    output_last = ''
    for piece in pieces:
        index_x = index % BOARD_SIZE
        index_y = (index // BOARD_SIZE) + 1
        piece_img = load_image(piece, IMG_SIZE_KERAS, PRE_INPUT_KERAS)
        resut_layer = model.predict(piece_img)[0].tolist()
        index_piece = resut_layer.index(max(resut_layer))
        output_first += (LETTERS[index_x] + str(index_y) + ' ' + __PREDS_DICT[index_piece] + ' ' + str(resut_layer)) + '\n'
        output_last += (LETTERS[index_x] + str(index_y) + ' ' + __PREDS_DICT[index_piece]) + '\n'
        index += 1
    print(output_first)
    print(output_last)