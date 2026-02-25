from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, models

from ind_detection.config import Config
from ind_detection.datasets.dataset import Dataset
from ind_detection.util.data_util import DataUtil


class Train(object):

    def __init__(self, model):
        # The path where the spreading space is structure encoded to store the dataset
        self.path_ref = Config.path_ref
        # Learning rate
        self.learning_rate = Config.learning_rate
        # Models that need to be trained
        self.model = model
        # Batch size
        self.batch_size = Config.batch_size
        # Training time
        self.epochs = Config.epochs
        # The dataset needed for the training or testing phase of the model
        self.train_db, self.test_db = None, None
        # The neural network Softmax layer
        self.softmax = tf.keras.layers.Softmax()

    def data_init(self):
        # Construct the dataset used for model training
        self.train_db = Dataset(mode='train')
        # Construct the dataset used for model testing
        self.test_db = Dataset(mode='test')

    def iteration(self, name, epoch):
        """
        Execution of an epoch procedure during model training or testing
        :param name: test or train
        :param epoch: epoch number
        :return:
        """
        # Get the dataset needed for training or testing the current epoch model
        data_train = self.train_db
        if 'test' == name:
            data_train = self.test_db

        # Get the neural network optimiser for the TensorFlow framework
        optimizer = optimizers.Adam(self.learning_rate)
        # define the variable values
        loss_num, acc, pred_matrix, label_li = 0, 0, None, []
        # Creating a terminal progress bar
        pbar = tqdm(total=data_train.len() // self.batch_size + 1)

        # Select GPU platform
        with tf.device('/gpu:0'):
            # Get all sample IDs
            for item in data_train.get_all():
                # load the sample data corresponding to the IDs into the memory as a batch size
                image_matrix, topic_label, diffuse_num, node_num = data_train.get_item(index_li=item)
                # Single training space for the TensorFlow framework
                with tf.GradientTape() as tape:
                    # The real labels [0,1,... ,1] for one-hot encoding [[1, 0], [0, 1],... ,[0, 1]]
                    label_matrix = tf.one_hot(topic_label, depth=2)
                    # Call the model to get the predicted data
                    out_matrix = self.model(image_matrix)
                    # Calculation of batch loss values using the cross-entropy function
                    loss = tf.losses.categorical_crossentropy(label_matrix, out_matrix, from_logits=True)
                    # Calculation of average loss values for batch samples
                    loss = tf.reduce_mean(loss)
                    # Normalization of predicted data using softmax function
                    out_matrix = self.softmax(out_matrix)
                    if 'train' == name:
                        # When the model is in the training phase,
                        # a gradient update of the model parameters based on the average loss value is required
                        grads = tape.gradient(loss, self.model.trainable_variables)
                        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                    # Save predictions for the current epoch model
                    if pred_matrix is None:
                        pred_matrix = out_matrix
                    else:
                        pred_matrix = tf.concat(values=[pred_matrix, out_matrix], axis=0)
                    # Save real labels for the current epoch model
                    [label_li.append(i) for i in topic_label]
                    # Save accuracy for the current epoch model
                    acc += DataUtil.acc(label_matrix=topic_label, out_matrix=out_matrix)
                    # Save loss sum for the current epoch model
                    loss_num += float(loss) * self.batch_size
                    # The terminal prints the current batch of training data
                    pbar.desc = f'epoch: {epoch}, name: {name}, loss: {round(loss_num / len(label_li), 4)}, accuracy: {round(acc / len(label_li), 4)}'
                    pbar.update(1)

        if Config.show_all:
            # Determine whether to print the data of the whole process after the current epoch has finished executing
            label_matrix = tf.one_hot(label_li, depth=2)
            accuracy = DataUtil.acc(label_matrix=label_li, out_matrix=pred_matrix)
            precision, recall, f1_score, macro_f1, micro_f1 = DataUtil.evaluation(y_test=label_matrix,
                                                                                  y_predict=pred_matrix)
            acc, loss = accuracy / data_train.len(), loss_num / data_train.len()
            print(f'macro f1: {round(macro_f1, 4)}, micro f1: {round(micro_f1, 4)}')
            print(f'accuracy: {round(acc, 4)}, loss: {round(loss, 4)}')
            print(
                f'non-precision: {round(precision[0], 4)}, non-recall: {round(recall[0], 4)}, non-f1_score: {round(f1_score[0], 4)}')
            print(
                f'rumor-precision: {round(precision[1], 4)}, rumor-recall: {round(recall[1], 4)}, rumor-f1_score: {round(f1_score[1], 4)}')
        else:
            acc = round(acc / len(label_li), 4)
        return acc

    def train(self):
        """
        Main functions for the model training and testing process
        :return:
        """
        # Dataset initialization, i.e. construction of training and test sets
        self.data_init()
        # Model training and testing process for each epoch
        for epoch in range(self.epochs):
            # Model training process
            self.iteration(name='train', epoch=epoch)
            # Model testing process
            self.iteration(name='test', epoch=epoch)
