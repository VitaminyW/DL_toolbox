"""
Author: Chen Yewei
From: Xiamen University
Github: https://github.com/VitaminyW
If you want to use this code please cite this repository.
Email: 1779723554@qq.com
"""

import requests


class BarkReminder:
    def __init__(self, bark_token: str, clock_timer: int, *, group_name=None, isArchive=True, level='active'):
        """
        :param bark_token: bark 的 token 用于发送服务器
        :param clock_timer: 定时器的间隔时间
        :param group_name: 消息属于的群组
        :param isArchive: 是否保存消息
        :param level: 消息通知等级 ['active':立即亮屏显示，'timeSensitive':可在专注模式下通知, 'passive': 仅添加到通知列表]
        """
        self.bark_token = bark_token
        self.clock_timer = clock_timer
        self.cur_timer = clock_timer
        self.group_name = group_name
        self.isArchive = isArchive
        self.level = level
        self.default_url = 'https://api.day.app/' + self.bark_token + '/'
        assert self.level in ['active', 'timeSensitive', 'passive'] and type(self.isArchive) == bool

    def remind(self, notice_dict: dict, title: str):
        """
        :param title: 通知标题
        :param notice_dict: 通知内容，将会以 key + value + /n 的形式结合
        :return: None
        """
        if self.cur_timer == 0:
            notice_url = self.default_url + title + '/'
            content = '\n'.join([f'{item[0]}:{str(item[1])}' for item in notice_dict.items()])
            notice_url = notice_url + content
            notice_url = notice_url + '?'
            if self.group_name is not None:
                notice_url = notice_url + 'group=' + self.group_name
            notice_url = notice_url + '&' + 'level=' + self.level + '&' + 'isArchive=' + str(int(
                self.isArchive))
            # print(notice_url)
            requests.get(notice_url)
            self.cur_timer = self.clock_timer
        else:
            self.cur_timer -= 1
    
    def __call__(self,notice_dict: dict, title: str):
        self.remind(notice_dict, title)
            
def get_reminder(bark_token,clock_timer,group_name):
    """
    :param bark_token: bark 的 token 用于发送服务器
    :param clock_timer: 定时器的间隔时间
    :param group_name: 消息属于的群组
    """
    return BarkReminder(bark_token,clock_timer,group_name=group_name)