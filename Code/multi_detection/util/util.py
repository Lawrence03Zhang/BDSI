import os

from multi_detection.config import Config


class Util(object):

    @staticmethod
    def save_metric(train_li, test_li):
        '''
        Save evaluation metrics from model training or testing process
        '''
        save_path = f'{Config.save_path}'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_file_path = ''
        index = -1
        for i in range(10000):
            save_file_path = f'{save_path}train{i}.txt'
            index = i
            if not os.path.exists(save_file_path):
                break
        with open(save_file_path, 'w+', encoding='utf-8') as f:
            for line in train_li:
                f.write(f'{line}\n')
        save_file_path = f'{save_path}test{index}.txt'
        with open(save_file_path, 'w+', encoding='utf-8') as f:
            for line in test_li:
                f.write(f'{line}\n')
