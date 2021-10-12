import os

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input as \
        prein_xception


from lc2fen.test_predict_board import predict_board
from lc2fen.predict_board import load_image

MODEL_PATH_KERAS = "/src/selected_models/alexnet.h5"
IMG_SIZE_KERAS = 224
PRE_INPUT_KERAS = prein_xception

def main_keras():
    """Executes Keras test board predictions."""
    print("Keras predictions")
    model = load_model(MODEL_PATH_KERAS)

    def obtain_pieces_probs(pieces):
        predictions = []
        for piece in pieces:
            piece_img = load_image(piece, IMG_SIZE_KERAS, PRE_INPUT_KERAS)
            predictions.append(model.predict(piece_img)[0])
        return predictions

    fen = predict_board(os.path.join("predictions", "test5.jpg"), "BR",
                        obtain_pieces_probs)

    print(fen)


if __name__ == "__main__":
    main_keras()