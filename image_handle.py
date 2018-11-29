import numpy as np 
import sys
import cv2 as cv 
import random

# ["critial","error","warning","info","debug"]
from Logging_Handle import *

#图片处理
class Image_handle(object):

    def __init__(self,image_path):
        logger.writeLog('CODE ENVIRONMENT:'+sys.version,level='info')
        try:
            self.image = cv.imread(image_path,1)
            self.image_heigth = self.image.shape[0]
            self.image_width = self.image.shape[1]
        except AttributeError as e:
            logger.writeLog('The image is no longer available',level='critial')
            sys.exit(e)
        except Exception as e:
            logger.writeLog('The image open failed',level='critial')
            sys.exit(e)

    #显示图片及其基本信息
    def image_show(self,image_value):

        #图片保存
        # img_path = 'Salt_filter.jpg'
        # cv.imwrite(img_path,image_value)
        logger.writeLog('width:'+str(image_value.shape[1])+' '*3+'heigth:'+str(image_value.shape[0]),level='info')
        cv.imshow('img',image_value)        #显示目标
        cv.imshow('original',self.image)    #显示原画
        cv.waitKey()

    #彩色直方图均衡化
    #直方图：亮度在不同区域分布的密度
    def Color_equalization(self):
        # 方法一：拆分颜色通道，均衡化后重组
        (B,G,R) = cv.split(self.image) #cv.split  多通道分离
        B = cv.equalizeHist(B)
        G = cv.equalizeHist(G)
        R = cv.equalizeHist(R)
        new_image = cv.merge((B,G,R))  #cv.merge  多通道合并
        self.image_show(new_image)

        # 方法二：映射到YUV(亮度，色度，饱和度)，再转回BGR
        # new_image = cv.cvtColor(self.image,cv.COLOR_BGR2YUV)
        # new_image[...,1] = cv.equalizeHist(new_image[...,1])
        # new_image = cv.cvtColor(new_image,cv.COLOR_YUV2BGR)
        # self.image_show(new_image)

    #椒盐滤波
    #随机更改像素色彩为255(盐点)或0(椒点)
    def Salt_filter(self,n):
        new_image = self.image.copy()   #深拷贝原图数据
        for i in range(n):
            x = int(np.random.random() * self.image_width)  #在原尺寸范围内生成随机坐标(int类型)
            y = int(np.random.random() * self.image_heigth)
            if new_image.ndim == 2:                     #二维数组，灰度图像
                new_image[y,x] = 255
            elif new_image.ndim == 3:
                new_image[y,x,0] = 255                  #三维数组，彩色图像，每个颜色通道255，合成盐点
                new_image[y,x,1] = 255
                new_image[y,x,2] = 255
        self.image_show(new_image)

    #中值滤波
    #将图像的每个像素用邻域像素的中值代替
    #滤波器大小一般为奇数，由于其会忽略最大最小值，所以中值滤波对椒盐噪声很有用
    def Median_filter(self):
        new_image = cv.medianBlur(self.image,5)     #中值滤波函数，原图像矩阵，滤波器大小(范围)(5*5)
        self.image_show(new_image) 

    #均值滤波
    #类似于中值滤波
    def Mean_filter(self):
        new_image = cv.blur(self.image,(5,5))   #滤波器大小为5*5
        self.image_show(new_image)

    #高斯滤波
    def Gaussian_filter(self):
        new_image = cv.GaussianBlur(self.image,(5,5),0) #(5,5):高斯核大小，0：高斯核在X方向的标准差
        self.image_show(new_image)

    #双边滤波
    #双边滤波是一种非线性的滤波方法，结合图像的空间临近度和像素值相似度
    #同时考虑空间与信息和灰度相似性，达到保边去噪的目的，具有简单、非迭代、局部处理的特点
    #滤波器由两个函数构成：一个函数是由几何空间距离决定滤波器系数，另一个是由像素差值决定滤波器系数
    # src：输入图像
    # d：过滤时周围每个像素领域的直径
    # sigmaColor：在color space中过滤sigma。参数越大，临近像素将会在越远的地方mix。
    # sigmaSpace：在coordinate space中过滤sigma。参数越大，那些颜色足够相近的的颜色的影响越大。
    def Double_filter(self):
        new_image = cv.bilateralFilter(self.image,9,75,75)
        self.image_show(new_image)

    #图片旋转
    def Image_rota(self,angle):
        #getRotationMatrix2D(选装的中心点,旋转的角度，图像缩放因子),主要用于获得图像关于某一点旋转的矩阵
        rota_matrix = cv.getRotationMatrix2D((self.image_width/2,self.image_heigth/2),angle,1)
        #warpAffine(需要图像,变换矩阵，变换后的图像大小)，仿射函数
        new_image = cv.warpAffine(self.image,rota_matrix,(self.image_width,self.image_heigth))
        self.image_show(new_image)

    #图片透明(>100:提升亮度)
    #addWeighted(输入图片1矩阵，融合比例，输入图片2矩阵，融合比例，偏差，输出阵列的可选深度(默认-1))
    #点运算函数，用来实现图片合成
    def Alpha(self,percent):
        new_image = cv.addWeighted(self.image,percent/100,self.image,0,0)
        self.image_show(new_image)

    #图片缩放
    #resize(输出图片，输出图像大小，width方向的缩放比例，height方向的缩放比例，差值方式(默认双线性差值))
    def Resize(self,fx,fy):
        new_image = cv.resize(self.image,None,fx=fx,fy=fy)
        self.image_show(new_image)      

if __name__ == '__main__':
    logger = Logging_Handle()
    image = Image_handle('/home/nan/桌面/OpenCv/Image_Handle/original.jpg')
    # image.Color_equalization()
    # image.Salt_filter(5000)
    # image.Median_filter()
    # image.Mean_filter()
    # image.Gaussian_filter()
    # image.Double_filter()
    # image.Image_rota(180)
    # image.Alpha(150)
    # image.Resize(2,1.5)

    logger.removeLog()