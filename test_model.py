from train_model import train_model


def test_model_training():
    model, accuracy = train_model()

    # Check that the model was trained
    assert model is not None

    # Check that accuracy is within a valid range
    assert 0 <= accuracy <= 1

    # Check that the model file was saved
    import os
    assert os.path.exists("artifacts/iris_model.pkl")