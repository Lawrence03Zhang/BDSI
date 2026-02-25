from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, models

from multi_detection.config import Config
from multi_detection.datasets.dataset import Dataset
from multi_detection.util.data_util import DataUtil
from multi_detection.util.time_util import time_util
from multi_detection.util.util import Util


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

    def iteration(self, name, epoch, train_li, test_li):
        """
        Execution of an epoch procedure during model training or testing
        Each line of code is interpreted as if it were the same file in the ‘ind_detection’ folder.
        Therefore, there is no more explanation.
        :param name: test or train
        :param epoch: epoch number
        :return:
        """
        data_train = self.train_db
        if 'test' == name:
            data_train = self.test_db
        optimizer = optimizers.Adam(self.learning_rate)
        loss_num, acc, diffuse_total, node_total = 0, 0, 0, 0
        pred_matrix, label_li = None, []
        start_time = int(time_util.os_stamp())
        pbar = tqdm(total=data_train.len() // self.batch_size + 1)
        with tf.device('/gpu:0'):
            for item in data_train.get_all():
                image_matrix, topic_label, diffuse_num, node_num = data_train.get_item(index_li=item)
                diffuse_total += diffuse_num
                node_total += node_num
                with tf.GradientTape() as tape:
                    label_matrix = tf.one_hot(topic_label, depth=2)
                    out_matrix = self.model(image_matrix)
                    loss = tf.losses.categorical_crossentropy(label_matrix, out_matrix, from_logits=True)
                    loss = tf.reduce_mean(loss)
                    loss_num += float(loss) * self.batch_size
                    out_matrix = self.softmax(out_matrix)
                    if pred_matrix is None:
                        pred_matrix = out_matrix
                    else:
                        pred_matrix = tf.concat(values=[pred_matrix, out_matrix], axis=0)
                    [label_li.append(i) for i in topic_label]
                    accuracy = DataUtil.acc(label_matrix=topic_label, out_matrix=out_matrix)
                    acc += accuracy
                    end_time = (int(time_util.os_stamp()) - start_time) / len(label_li)
                    pr_auc = DataUtil.pr_auc(y_label=label_li, y_predict=pred_matrix)
                    desc = f'epoch: {epoch}, name: {name}, loss: {round(loss_num / len(label_li), 4)}, accuracy: {round(acc / len(label_li), 4)}'
                    desc += f', auc: {round(pr_auc, 4)}, relation: {round(diffuse_total / node_total, 4)}, time: {end_time} ms'
                    pbar.desc = desc
                    pbar.update(1)
                    if 'train' == name:
                        train_li.append(desc)
                        grads = tape.gradient(loss, self.model.trainable_variables)
                        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                    else:
                        test_li.append(desc)
        pr_auc = DataUtil.pr_auc(y_label=label_li, y_predict=pred_matrix)
        if Config.show_all:
            label_matrix = tf.one_hot(label_li, depth=2)
            del tape
            hot_label_li, hot_pred_li = DataUtil.list2_list1(list2=label_matrix), DataUtil.list2_list1(
                list2=pred_matrix)
            auc_score = DataUtil.auc_compute(real_li=hot_label_li, pred_li=hot_pred_li)
            accuracy = DataUtil.acc(label_matrix=label_li, out_matrix=pred_matrix)
            precision, recall, f1_score, macro_f1, micro_f1 = DataUtil.evaluation(y_test=label_matrix,
                                                                                  y_predict=pred_matrix)
            acc, loss = accuracy / data_train.len(), loss_num / data_train.len()
            print(f'auc score: {round(auc_score, 4)}, macro f1: {round(macro_f1, 4)}, micro f1: {round(micro_f1, 4)}')
            print(f'accuracy: {round(acc, 4)}, loss: {round(loss, 4)}')
            print(
                f'non-precision: {round(precision[0], 4)}, non-recall: {round(recall[0], 4)}, non-f1_score: {round(f1_score[0], 4)}')
            print(
                f'rumor-precision: {round(precision[1], 4)}, rumor-recall: {round(recall[1], 4)}, rumor-f1_score: {round(f1_score[1], 4)}')
        else:
            acc = round(acc / len(label_li), 4)
        return acc, pr_auc

    def train(self):
        """
        Main functions for the model training and testing process
        :return:
        """
        # Dataset initialization, i.e. construction of training and test sets
        self.data_init()
        train_li, test_li = [], []
        acc_max = -1
        # Model training and testing process for each epoch
        for epoch in range(self.epochs):
            # Model training process
            self.iteration(name='train', epoch=epoch, train_li=train_li, test_li=test_li)
            # Model testing process
            acc, pr_auc = self.iteration(name='test', epoch=epoch, train_li=train_li, test_li=test_li)
            if pr_auc > acc_max:
                acc_max = pr_auc
        print(f'max acc: {acc_max}')
        # save data
        Util.save_metric(train_li=train_li, test_li=test_li)
        return acc_max
