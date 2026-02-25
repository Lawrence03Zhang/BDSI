import json
import numpy as np

from ind_detection.config import Config


class ImageUtil(object):

    @staticmethod
    def load_json(path):
        """
        load json data
        """
        with open(path, encoding='utf-8') as f:
            data_json = json.load(f)
        return data_json

    @staticmethod
    def cutting(pixel_li, area_num=Config.area_num):
        """
        cutting algorithm
        :param pixel_li: pixel list
        :param area_num: grid idle rate
        """
        x_li, y_li = [i[0] for i in pixel_li], [i[1] for i in pixel_li]
        x_d, y_d = max(x_li) - min(x_li), max(y_li) - min(y_li)
        pixel_area = x_d * y_d / len(pixel_li) / area_num
        pixel_d = pixel_area ** 0.5
        return pixel_d, min(x_li), min(y_li)

    @staticmethod
    def build_init_image(image_size=Config.image_size):
        """
        build init image-like
        """
        init_image = np.zeros(shape=image_size, dtype=np.int64)
        return init_image

    @staticmethod
    def compute_index(temp, cut_dis, szie):
        """
        Calculate the index of the current node in grid space
        """
        index = temp // cut_dis
        if temp % cut_dis == 0:
            return_index = index
        else:
            return_index = index + 1
        if return_index >= szie:
            return_index = szie - 1
        return int(return_index)

    @staticmethod
    def diffuse(x, y, cut_dis, x_min, y_min, layer, init_image, num):
        """
        Diffuse algorithm
        """
        x_size, y_size = len(init_image), len(init_image[0])
        x_index = ImageUtil.compute_index(temp=x - x_min, cut_dis=cut_dis, szie=x_size)
        y_index = ImageUtil.compute_index(temp=y - y_min, cut_dis=cut_dis, szie=y_size)
        if sum(init_image[x_index][y_index]) == 0:
            return x_index, y_index, num
        num += 1
        i = max(x_index - layer, 0)
        for j in range(max(0, y_index - layer), min(y_size, y_index + layer + 1), 1):
            if sum(init_image[i][j]) == 0:
                return i, j, num
        j = min(y_index + layer, y_size - 1)
        for i in range(max(0, x_index - layer), min(x_size, x_index + layer + 1), 1):
            if sum(init_image[i][j]) == 0:
                return i, j, num
        i = min(x_index + layer, x_size - 1)
        for j in range(min(y_size - 1, y_index + layer), max(-1, y_index - layer - 1), -1):
            if sum(init_image[i][j]) == 0:
                return i, j, num
        j = max(x_index - layer, 0)
        for i in range(min(x_size - 1, y_index + layer), max(-1, x_index - layer - 1), -1):
            if sum(init_image[i][j]) == 0:
                return i, j, num
        return -1, -1, num

    @staticmethod
    def load_image(path):
        """
        Main function of the G2I algorithm
        :param path: 2D space save path (after ... /... /datasets/image_like/structure_encoding.py)
        :return: generated image-like data
        """
        pixel_li = []
        with open(path, encoding='utf-8') as f:
            line = f.readline()
            while line:
                att_li = line.replace('\n', '').split(',')
                x, y = float(att_li[0]), float(att_li[1])
                r, g, b = [int(i) for i in att_li[-1].split(' ') if i != '']
                pixel_li.append([x, y, r, g, b])
                line = f.readline()
        image_like = ImageUtil.image_generation(pixel_li=pixel_li)
        return image_like

    @staticmethod
    def image_generation(pixel_li):
        """
        image-like generation process
        :param pixel_li: node pixel list
        :return:
        """
        init_image = ImageUtil.build_init_image()
        x_size, y_size = len(init_image), len(init_image[0])
        if x_size * y_size <= len(pixel_li):
            raise RuntimeError(f'网格数{x_size * y_size}不够节点{len(pixel_li)}分配')
        cut_dis, x_min, y_min = ImageUtil.cutting(pixel_li=pixel_li)
        diffuse_total = 0
        for x, y, r, g, b in pixel_li:
            i, j, layer, diffuse_num = -1, -1, 1, 0
            while i == -1 or j == -1:
                i, j, diffuse_num = ImageUtil.diffuse(x=x, y=y, cut_dis=cut_dis, x_min=x_min, y_min=y_min, layer=layer,
                                                      init_image=init_image, num=diffuse_num)
                layer = layer + 1
            if diffuse_num > 2:
                diffuse_total += 1
            init_image[i][j] = [r, g, b]
        return init_image, diffuse_total, len(pixel_li)
