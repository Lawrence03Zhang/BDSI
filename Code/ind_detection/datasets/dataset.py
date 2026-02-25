import json

import numpy as np
import tensorflow as tf

from ind_detection.config import Config
from ind_detection.util.image_util import ImageUtil


class Dataset(object):
    """
    Class for dataset loading and construction
    Due to the huge size of the dataset, it is not possible to put the complete data into memory in a uniform way.
    Therefore, a dynamic build-and-load strategy is used.
    1) construction:
        All datasets are saved locally, the save path can be modified in Config.py.
        train.json and test.json files save the data as
        { ‘sample id’: {
                ‘path": “non/7395250353”,
                ‘label": 0
            },
            ‘sample id": {
                ‘path": “non/7764175917”,
                ‘label": 0
            }
        }
        Where path is the real sample path and label is the real sample label.
    2) loading:
        The dataset is divided into batches based on sample IDs.
        The required sample IDs are obtained for each training or test,
        and the local data is subsequently loaded.
    """
    def __init__(self, mode):
        # Batch size
        self.batch_size = Config.batch_size
        # Get the train.json or test.json file
        if 'train' == mode:
            self.data = json.load(open(f'{Config.path_ref}/train.json'))
        else:
            self.data = json.load(open(f'{Config.path_ref}/test.json'))

    def process_index(self, index):
        """
        Converting python int types to types recognised by the TensorFlow framework
        :param index: Sample ID of type int
        :return: Sample ID of type TensorFlow
        """
        index = tf.cast(index, tf.int32)
        return index

    def get_all(self):
        """
        All sample ids are given to the TensorFlow framework for random batch segmentation.
        :return: sample ids datasets
        """
        index_li = np.asarray([int(i) for i in self.data])
        data_db = tf.data.Dataset.from_tensor_slices(index_li)
        data_db = data_db.map(self.process_index).shuffle(10000).batch(self.batch_size)
        return data_db

    def get_item(self, index_li):
        """
        Load local data into memory based on the sample id of the current batch
        :param index_li: sample id of the current batch
        :return: local data
        """
        # Converting a TensorFlow type sample id to an int type
        index_li = index_li.numpy()
        # Define variable values
        image_li, topic_label, diffuse_total, node_total = [], [], 0, 0
        for index in index_li:
            # Load local data for the current sample id
            image_like, label, diffuse_num, node_num = self.iteration(index=index)

            # Save local data
            image_li.append(image_like)
            topic_label.append(label)
            diffuse_total += diffuse_num
            node_total += node_num

        # Converting data types into ones that the TensorFlow framework can handle
        image_matrix = np.asarray(image_li, dtype=np.float32)
        topic_label = np.asarray(topic_label, dtype=np.int32)
        return image_matrix, topic_label, diffuse_total, node_total

    def iteration(self, index):
        """
        Load local data for the current sample id
        :param index: current sample id
        :return: local data
        """
        # Sample id of type int converted to str
        index = str(index)
        # Construct a full save path for local data based on sample ids
        path = Config.path_ref + self.data[index]['path']
        # Load local data based on full save path
        image_like, diffuse_total, node_total = ImageUtil.load_image(path=path)
        # Load real label
        label = self.data[index]['label']
        return image_like, label, diffuse_total, node_total

    def len(self):
        """
        Number of samples obtained
        """
        return len(self.data.keys())
