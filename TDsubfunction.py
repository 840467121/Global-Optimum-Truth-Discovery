# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 08:40:32 2021

@author: 84046
"""
import math
import numpy as np
from numpy import *
import pandas as pd
from sklearn.cluster import KMeans
import scipy.spatial.distance as dist 
import TDsubfunction as sub



def calcuDistance(data1, data2):
    '''
    计算两个模式样本之间的欧式距离
    :param data1:
    :param data2:
    :return:
    '''
    distance = 0
    for i in range(len(data1)):
        distance += pow((data1[i]-data2[i]), 2)
    return math.sqrt(distance)


#使用jaccard距离
def distanceJ(p1, p2):
    res = 0.0
    Pd = []
    Pd.append(p1)
    Pd.append(p2)
    res = dist.pdist(Pd,'jaccard') 
    return res

#使用欧式距离
def distanceO(p1, p2):
    res = 0.0
    Pd = []
    # if(p1!=p2):
    Pd.append(p1)
    Pd.append(p2)
    for h in range(len(p1)): 
        res += np.linalg.norm(p1[h]-p2[h])
    return res

def entropy(pstar, P, n):
    n = 50
    pentropy = []
    for i in range (0,len(pstar)):
      Pentropy = 0.0
      for j in range (0,n):
          if P[j][i]-pstar[i] == 0 :
              Pentropy = Pentropy
          else:
              Pentropy += -math.pow(np.linalg.norm(P[j][i]-pstar[i]),2)*math.log(math.pow(np.linalg.norm(P[j][i]-pstar[i]),2))
      pentropy.append(Pentropy)  
      return np.array(pentropy)

def Delta(P,x): #计算求k时所需的Δ，
   distance = []
   n = len(P)
   if (x == 1):     
       for i in  range(0,n-2) :
            for j in range(i+1,n-1) :          
                     p1=P[i]
                     p2=P[j]                         
                     distance.append(sub.distanceJ(p1,p2))  
   else :
       for i in  range(0,n-2) :
            for j in range(i+1,n-1) :          
                    p1=P[i]
                    p2=P[j]                         
                    distance.append(sub.calcuDistance(p1,p2)) 
   delta=max(distance)/min(distance)
   print(max(distance),min(distance),max(distance)/min(distance))  
   return delta

def weights(pStar, P, n): #计算权重 （论文p20 2-5）
    sum_result = 0.0
    W = []
    for i in range(0, n):
        sum_result += math.pow(np.linalg.norm(pStar - P[i]), 2)
        #sum_result += math.pow(distance(pStar, P[i]), 2)
        #print (i)
    for l in range(0, n):
        if((pStar==P[l]).all()):
            w = 3
            W.append(w)
        else:
            w = math.log10(sum_result / math.pow(np.linalg.norm(pStar - P[l]), 2))
            #w = math.log10(sum_result / math.pow(distance(pStar, P[i]), 2))
            W.append(w)
    return W

def objectFunction(pStar, W, P, n): #论文 （2-4）
    object = []
    for i in range(0, n):
        w = W[i]
        p = math.pow(np.linalg.norm(pStar-P[i]),2)
        #p = math.pow(distance(pStar,P[i]),2)
        ##print type(p), type(w)
        s = float(w)*float(p)
        object.append(s)
    os = 0.0
    for i in range(0, n):
        os += object[i]
    return os #目标函数的其中一种，还不是最小值

def half(p1, p2):
    return (p1+p2)/2


def maxmin_distance_cluster(data, Theta):
    '''
    :param data: 输入样本数据,每行一个特征
    :param Theta:阈值，一般设置为0.5，阈值越小聚类中心越多
    :return:样本分类，聚类中心
    '''
    maxDistance = 0
    start = 0#初始选一个中心点
    index = start#相当于指针指示新中心点的位置
    k = 0 #中心点计数，也即是类别
 
    dataNum=len(data)
    distance=np.zeros((dataNum,))
    minDistance=np.zeros((dataNum,))
    classes =np.zeros((dataNum,))
    centerIndex=[index]
 
    # 初始选择第一个为聚类中心点
    ptrCen=data[0]
    # 寻找第二个聚类中心，即与第一个聚类中心最大距离的样本点
    for i in range(dataNum):
        ptr1 =data[i]
        d=distanceJ(ptr1,ptrCen)#####距离
        distance[i] = d
        classes[i] = k + 1
        if (maxDistance < d):
            maxDistance = d
            index = i #与第一个聚类中心距离最大的样本
 
    minDistance=distance.copy()
    maxVal = maxDistance
    while maxVal > (maxDistance * Theta):
        k = k + 1
        centerIndex+=[index] #新的聚类中心
        for i in range(dataNum):
            ptr1 = data[i]
            ptrCen=data[centerIndex[k]]
            d = distanceJ(ptr1, ptrCen)#####距离
            distance[i] = d
            #按照当前最近临方式分类，哪个近就分哪个类别
            if minDistance[i] > distance[i]:
                minDistance[i] = distance[i]
                classes[i] = k + 1
        # 寻找minDistance中的最大距离，若maxVal > (maxDistance * Theta)，则说明存在下一个聚类中心
        index=np.argmax(minDistance)
        maxVal=minDistance[index]
    return classes,centerIndex
 
def  GetSegmentPoint(nLayerNo,nPointIndex,pointindex,OriginalPoint,p,bIs):
    # nLayerNo = 2
    # nPointIndex = 0
    if not bIs:
          return
    n = len(pointindex);
    if (nLayerNo < 0 or nLayerNo > n):
         bIs= False
         return
    if (nPointIndex < 0 or nPointIndex >= len(pointindex[nLayerNo-1])):
         bIs= False
         return
    #设数组Layer_PointIndex保存某一层点的个数
    Layer_PointIndex = pointindex[nLayerNo-1]
    
    if (len(Layer_PointIndex)==0):
         bIs= False
         return
    # Point p1,p2;     
    if (Layer_PointIndex[nPointIndex][0] == 0): #如果是原始点层，则可以获得此点
         p1 = OriginalPoint[Layer_PointIndex[nPointIndex][1]]
         #print (Layer_PointIndex[nPointIndex][0],Layer_PointIndex[nPointIndex][1])
         bIs = True
    else:
        p1 = 0 
        #print (Layer_PointIndex[nPointIndex][0],Layer_PointIndex[nPointIndex][1])
        p1 = GetSegmentPoint(Layer_PointIndex[nPointIndex][0],Layer_PointIndex[nPointIndex][1],pointindex,OriginalPoint,p1,bIs)
    if not bIs:
         return   
    if (Layer_PointIndex[nPointIndex][2] == 0): #如果是原始点层，则可以获得此点
         p2 = OriginalPoint[Layer_PointIndex[nPointIndex][3]]
        # print (Layer_PointIndex[nPointIndex][2],Layer_PointIndex[nPointIndex][3])
         bIs = True
    else:
        p2 = 0
       # print (Layer_PointIndex[nPointIndex][2],Layer_PointIndex[nPointIndex][3]) 
        p2 = GetSegmentPoint(Layer_PointIndex[nPointIndex][2],Layer_PointIndex[nPointIndex][3],pointindex,OriginalPoint,p2,bIs)
    if not bIs:
          return	     
    p = (p1 + p2) / 2;    
    return p


