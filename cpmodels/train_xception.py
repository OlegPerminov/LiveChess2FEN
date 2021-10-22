"""
Train Xception model.
"""
from tensorflow import keras
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.models import load_model

from chess_piece_models_common import build_model, data_generators, \
    train_model, plot_model_history, evaluate_model, model_callbacks


BATCH_SIZE = 2
IMAGE_SIZE = 299
WORKERS = 2

def train_chesspiece_model():
    """Trains the chesspiece model based on Xception."""
    base_model = Xception(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False,
                          weights='imagenet')

    # First train only the top layers
    for layer in base_model.layers:
        layer.trainable = False

    model = build_model(base_model)

    train_generator, validation_generator = data_generators(
        preprocess_input, (IMAGE_SIZE, IMAGE_SIZE), BATCH_SIZE)

    callbacks = model_callbacks(5, "./models/Xception_pre.h5", 0.1, 10)

    history = train_model(model, 20, train_generator, validation_generator,
                          callbacks, use_weights=False, workers=WORKERS)

    plot_model_history(history, "./models/Xception_pre_acc.png",
                       "./models/Xception_pre_loss.png")
    evaluate_model(model, validation_generator)

    # Also train blocks 12-14
    for layer in model.layers[:106]:
        layer.trainable = False
    for layer in model.layers[106:]:
        layer.trainable = True

    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = model_callbacks(20, "./models/Xception.h5", 0.2, 8)

    history = train_model(model, 100, train_generator, validation_generator,
                          callbacks, use_weights=False, workers=5)

    plot_model_history(history, "./models/Xception_acc.png",
                       "./models/Xception_loss.png")
    evaluate_model(model, validation_generator)

    model.save("./models/Xception_last.h5")


def continue_training():
    """Continues training the chesspiece model based on Xception."""
    model = load_model("./models/Xception.h5")

    train_generator, validation_generator = data_generators(
        preprocess_input, (IMAGE_SIZE, IMAGE_SIZE), BATCH_SIZE)

    # Train all layers
    for layer in model.layers:
        layer.trainable = True

    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = model_callbacks(20, "./models/Xception_all.h5", 0.2, 8)

    history = train_model(model, 100, train_generator, validation_generator,
                          callbacks, use_weights=False, workers=WORKERS)

    plot_model_history(history, "./models/Xception_all_acc.png",
                       "./models/Xception_all_loss.png")
    evaluate_model(model, validation_generator)

    model.save("./models/Xception_all_last.h5")


if __name__ == "__main__":
    #train_chesspiece_model()
    continue_training()
