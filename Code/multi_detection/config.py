class Config(object):
    # grid idle rate
    area_num = 1.6
    # Name of training dataset and save
    dataset_name = 'Dataset_2D/'
    # The path where the spreading space is structure encoded to store the dataset
    path_ref = f'CVPR/multi/weibo/{dataset_name}'
    # The path to save training data
    save_path = f'data/multi/{dataset_name}'
    # Standard image-like size
    image_size = (200, 200, 3)
    # Batch size
    batch_size = 8
    # Learning rate
    learning_rate = 0.001
    # Max size of historical user behaviour
    behavior_size = 100000
    # Behavioural similarity scores
    sim_num = 0.999
    # Training time
    epochs = 200
    # Dimension of the model output vector
    out_size = 2
    # Whether information on evaluation indicators is printed on the console
    show_all = False
