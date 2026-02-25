import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, auc


class DataUtil(object):

    @staticmethod
    def max_index(pred_li):
        """
        Calculate the index of the largest value in a list
        :param pred_li: list
        :return: index
        """
        index = 0
        temp = -1000
        for i, num in enumerate(pred_li):
            if num > temp:
                temp = num
                index = i
        return index

    @staticmethod
    def acc(label_matrix, out_matrix):
        """
        Calculating the accuracy of a binary classification task
        :param label_matrix: real label value
        :param out_matrix: Model predictions
        :return: Accuracy
        """
        out_matrix = out_matrix.numpy()
        true_total = 0
        for i in range(len(label_matrix)):
            if label_matrix[i] == -1:
                continue
            pred = tf.nn.softmax(out_matrix[i])
            pred_index = DataUtil.max_index(pred_li=pred)
            if pred_index == label_matrix[i]:
                true_total += 1
        return true_total

    @staticmethod
    def normal(predict_li, depth=3):
        """
        evaluation data processing algorithms in functions
        """
        result_li = []
        for att_li in predict_li.numpy():
            index = DataUtil.max_index(att_li)
            result_li.append(index)
        result_li = np.asarray(result_li, dtype=np.int32)
        result_li = tf.cast(result_li, dtype=tf.int32)
        result_li = tf.one_hot(result_li, depth=depth)
        return result_li

    @staticmethod
    def evaluation(y_test, y_predict):
        """
        Calculation of classification evaluation metrics (precision, recall and F1 value)
        :param y_test: real label value
        :param y_predict: Model predictions
        :return: Evaluation metrics
        """
        y_predict = DataUtil.normal(predict_li=y_predict, depth=2)
        metrics = classification_report(y_test, y_predict, output_dict=True)
        precision = metrics['0']['precision'], metrics['1']['precision']
        recall = metrics['0']['recall'], metrics['1']['recall']
        f1_score = metrics['0']['f1-score'], metrics['1']['f1-score']
        macro_f1 = metrics['macro avg']['f1-score']
        micro_f1 = metrics['micro avg']['f1-score']
        return precision, recall, f1_score, macro_f1, micro_f1

    @staticmethod
    def un_depth(data_li, label_li):
        """
        Multi-category labelling for levelling
        """
        return_li = []
        for i in range(len(list(data_li))):
            return_li.append(data_li[i][label_li[i]])
        return return_li

    @staticmethod
    def pr_auc(y_label, y_predict):
        """
        compute PR-AUC value
        :param pred_li: Model predictions
        :param real_li: real label value
        :return: PR-AUC value
        """
        y_predict = DataUtil.un_depth(data_li=y_predict, label_li=y_label)
        precision, recall, thresholds = precision_recall_curve(y_label, y_predict)
        pr_auc = auc(recall, precision)
        return pr_auc


    @staticmethod
    def auc_compute(pred_li, real_li):
        """
        compute AUC value
        :param pred_li: Model predictions
        :param real_li: real label value
        :return: AUC value
        """
        auc = roc_auc_score(y_true=real_li, y_score=pred_li)
        return auc

    @staticmethod
    def roc_compute(pred_li, real_li):
        """
        ROC curve
        :param pred_li: Model predictions
        :param real_li: real label value
        :return: FPR nad TPR
        """
        fpr, tpr, thresholds_keras = roc_curve(y_true=real_li, y_score=pred_li)
        return fpr, tpr