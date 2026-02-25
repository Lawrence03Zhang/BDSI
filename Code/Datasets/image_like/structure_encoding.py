import json

import networkx as nx
import numpy as np
from node2vec import Node2Vec
from sklearn.manifold import TSNE


class Util(object):

    @staticmethod
    def load(path):
        """
        load spreading space data
        """
        response_li = []
        with open(path, encoding='utf-8') as f:
            line = f.readline()
            while line:
                att_li = line.replace('\n', '').split('	')
                local_id, parent_id, pos, neu, neg = att_li[0], att_li[1], float(att_li[-4]), float(att_li[-3]), float(
                    att_li[-2])
                response_li.append((local_id, parent_id, pos, neu, neg))
                line = f.readline()
        return response_li

    @staticmethod
    def build_graph(response_li: list):
        """
        build graph from spreading space (using networkx)
        """
        entire_graph = nx.Graph()
        for (local_id, parent_id, pos, neu, neg) in response_li:
            entire_graph.add_edge(u_of_edge=local_id, v_of_edge=parent_id, weight=1)
            entire_graph.add_edge(u_of_edge=parent_id, v_of_edge=local_id, weight=1)
        return entire_graph

    @staticmethod
    def graph_to_vec(response_li):
        """
        random walk algorithm (using node2vec)
        """
        rel_graph = Util.build_graph(response_li=response_li)
        nodes = rel_graph.nodes
        model = Node2Vec(rel_graph, walk_length=5, num_walks=80, workers=4, p=0.8, q=0.5)
        mode = model.fit()
        value_list, key_list = [], []
        for node in nodes:
            predict_list = mode.wv.most_similar(str(node), topn=7)
            predict_list = [int(i[0]) for i in predict_list]
            value_list.append(predict_list)
            key_list.append(int(node))
        nodes = Util.t_sne(value_list=value_list, key_list=key_list)
        return nodes

    @staticmethod
    def t_sne(value_list: list, key_list: list):
        """
        T-SNE algorithm
        """
        x = np.array(value_list)
        ts = TSNE(n_components=2, perplexity=min(len(value_list) - 1, 30))
        ts.fit_transform(x)
        value_list = ts.embedding_
        nodes = dict()
        for i in range(len(key_list)):
            nodes[key_list[i]] = list(value_list[i])
        return nodes

    @staticmethod
    def node_to_pixel(response_li):
        """
        node to pixel function
        :param response_li: response node list
        :return: pixel list
        """
        node_dict = dict()
        response_li.append(('0', '0', 0, 0, 0))
        for (local_id, parent_id, pos, neu, neg) in response_li:
            pixel_rgb = [pos * 255, neu * 255, neg * 255]
            pixel_rgb = np.array(pixel_rgb, dtype=np.int64)
            node_dict[local_id] = pixel_rgb
        return node_dict

    @staticmethod
    def save_json(path, data_json):
        """
        json data save function
        :param path: save path
        :param data_json: json data
        """
        with open(path, 'w+', encoding='utf-8') as f:
            f.write(json.dumps(data_json, ensure_ascii=False))

    @staticmethod
    def graph_to_2dvec(path, save_path):
        """
        Structure encoding process main function
        :param path: User behaviour-driven spreading space repository address
        :param save_path: Encoded 2D space repository address
        """
        response_li = Util.load(path=path)
        node_dict = Util.node_to_pixel(response_li=response_li)
        nodes = Util.graph_to_vec(response_li=response_li)
        with open(f'{save_path}', 'w+', encoding='utf-8') as f:
            for node in nodes.keys():
                if str(node) not in node_dict.keys():
                    continue
                vec_li = nodes[node]
                rgb_li = node_dict[str(node)]
                pixel = str(vec_li)[1:-1] + ',' + str(rgb_li)[1:-1]
                f.write(f'{pixel}\n')
