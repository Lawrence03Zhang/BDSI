import datetime
import time


class time_util(object):

    @staticmethod
    def os_normal():
        """
        Get the current moment
        :return: current moment（2022-08-22 08:19:23）
        """
        return str(datetime.datetime.now()).split('.')[0]

    @staticmethod
    def os_stamp():
        """
        Get current timestamp
        :return: current timestamp（1661127563）
        """
        return str(time.time()).split('.')[0]

    @staticmethod
    def stamp_to_normal(time_stamp):
        """
        Timestamp type (1661127563) converted to standard time type (2022-08-22 08:19:23)
        :param time_stamp: Timestamp type（1661127563）
        :return: standard time type（2022-08-22 08:19:23）
        """
        time_stamp = time.localtime(time_stamp / 1000)
        return time.strftime("%Y-%m-%d %H:%M:%S", time_stamp)

    @staticmethod
    def weibo_to_normal(time_weibo, type='+0000'):
        """
        Weibo or Twitter time type (Mon Aug 22 08:19:23 +0800 2022) converted to standard (2022-08-22 08:19:23)
        :param time_weibo: Weibo or Twitter time type（Mon Aug 22 08:19:23 +0800 2022）
        :return: standard time type（2022-08-22 08:19:23）
        """
        return str(datetime.datetime.strptime(time_weibo, f'%a %b %d %H:%M:%S {type} %Y'))

    @staticmethod
    def normal_to_stamp(time_normal):
        """
        Standard time type (2022-08-22 08:19:23) converted to timestamp type (1661127563)
        :param time_normal: Standard time type（2022-08-22 08:19:23）
        :return: timestamp type（1661127563）
        """
        return int(time.mktime(time.strptime(time_normal, "%Y-%m-%d %H:%M:%S")))

    @staticmethod
    def weibo_convert(time_weibo, stamp=False):
        """
        Weibo or Twitter Time Type (Mon Aug 22 08:19:23 +0800 2022) Conversion Methods
        :param time_weibo: Weibo or Twitter Time Type（Mon Aug 22 08:19:23 +0800 2022）
        :param stamp: Is it converted to a timestamp type?（1661127563）. Default Conversion Standard Time Type（2022-08-22 08:19:23）
        :return: Standard (2022-08-22 08:19:23) or Timestamp type (1661127563)
        """
        time_result = time_util.weibo_to_normal(time_weibo)
        if stamp:
            time_result = time_util.normal_to_stamp(time_result)
        return time_result
