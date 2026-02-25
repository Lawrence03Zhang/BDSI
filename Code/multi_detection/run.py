from multi_detection.model.cnn import CNN
from multi_detection.train.train import Train


def main():
    """
    Main function for BDSI model training
    Control hyper-parameters can be found and changed in config.py
    """
    # To be trained BDSI_multi model
    BDSI_model = CNN()
    # Initialization of the model training process
    train = Train(model=BDSI_model)
    # Training model
    train.train()


if __name__ == '__main__':
    main()
