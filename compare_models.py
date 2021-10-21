import os

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input as \
        prein_xception


from lc2fen.test_predict_board import predict_board
from lc2fen.predict_board import load_image


MODEL_PARAMS = [
    ["/src/selected_models/AlexNet.h5", 224],
    ["/src/selected_models/DenseNet201_last.h5", 224],
    ["/src/selected_models/MobileNetV2_0p5_all.h5", 224],
    ["/src/selected_models/MobileNetV2_0p35_all_last.h5", 224],
    ["/src/selected_models/NASNetMobile_all_last.h5", 224],
    ["/src/selected_models/SqueezeNet1p1.h5", 227],
    ["/src/selected_models/Xception_last.h5", 299]
]
PRE_INPUT_KERAS = prein_xception

def main_keras():
    result = []
    for obj in MODEL_PARAMS:
        result.append(calc_fen(obj[0], obj[1]))

    for obj in result:
        print(obj[0], "\t\t", obj[1])

def calc_fen(model_path, img_size):
    """Executes Keras test board predictions."""
    print("Keras predictions")
    model = load_model(model_path)

    def obtain_pieces_probs(pieces):
        predictions = []
        for piece in pieces:
            piece_img = load_image(piece, img_size, PRE_INPUT_KERAS)
            predictions.append(model.predict(piece_img)[0])
        return predictions

    fen = predict_board(os.path.join("predictions", "chess3.jpg"), "BL",
                        obtain_pieces_probs)

    model_name = model_path.split('/')[-1]
    return [model_name, fen]

if __name__ == "__main__":
    main_keras()
