"""
Created on 
@author: 
"""

import logging 
import sys 

class Logging_Handle(object):

    def __init__(self,name='ROOT',setLever=logging.DEBUG):
        self.logger = logging.getLogger(name)   #获得日志对象
        self.logger.setLevel(setLever)          #设置日志级别，默认不记录debug和info
        self.formatter = logging.Formatter('%(asctime)s %(name)s- %(levelname)s - %(message)s')     #定义日志输出格式
        

        self.file_handler = logging.FileHandler(name+'.log')    #保存日志到文件
        self.file_handler.setLevel(setLever)                    #设置输出到文件的日志级别
        self.file_handler.setFormatter(self.formatter)          #选择保存的日志格式

        self.consle_handler = logging.StreamHandler()       #日志输出到屏幕控制台
        self.consle_handler.setLevel(setLever)              #设置输出到屏幕的日志级别
        self.consle_handler.setFormatter(self.formatter)    #选择输出的格式

        self.logger.addHandler(self.file_handler)        #添加文件日志
        self.logger.addHandler(self.consle_handler)      #添加输出日志

    def writeLog(self,info,level='debug'):
        if level == "critial":
            self.logger.critical(info)
        elif level == "error":
            self.logger.error(info)
        elif level == "warning":
            self.logger.warning(info)
        elif level == "info":
            self.logger.info(info)
        else:
            self.logger.debug(info)

    def removeLog(self):
        self.logger.removeHandler(self.file_handler)
        self.logger.removeHandler(self.consle_handler)

if __name__ == '__main__':
    logger = Logging_Handle()
    logger.writeLog('it is a big misstake',level='debug')
    logger.removeLog()

