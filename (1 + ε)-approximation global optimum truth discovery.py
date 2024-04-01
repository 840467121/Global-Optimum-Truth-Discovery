# -*- coding: utf-8 -*-
"""
Created on Tue May 31 10:52:11 2023

@author: 84046
"""
import time
start = time.perf_counter()

import random
import numpy as np 
import pandas as pd
import TDsubfunction as sub
import math
import matplotlib.pyplot as plt
import copy
import csv
import scipy.spatial.distance as dist 


global Plist
global Csum

P = pd.read_csv("50模型得分.csv").values
dataframe = pd.read_csv("50模型得分.csv",low_memory=False)
Plist = P.tolist()


##计算delta
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
delta1 = Delta(P,1)
delta2 = Delta(P,0)
###输入加一个重复点的检测


# dataframe = pd.read_excel(r'C:/Users/84046/Desktop/2021.12真知发现/kmeans.xlsx',
#                         sheet_name='Sheet4',
#                         usecols=[i for i in range (50)])
# dataframe.columns=["x", "y"]


# dataframe = pd.DataFrame([
#         [0, 0],[0,1],[1,1],[2,1],[3,3],[4,2],[-2,-1],[-1,2],
#         [1,0],[2,0],[3,0],[4,0],[5,0],[5,1],[5,2],[2,2],[3,4],
#         [10, 10],[10,13],[12,8],[12,11],[13,9],[14,6],[10,9],
#         [9,9],[9,8],[8,9],[11,11],[11,13],[12,10],[9,11],
#         [10, -10],[10,-9],[10,-13],[12,-10],[13,-13],[14,-12],
#         [9,-9],[9,-8],[8,-9],[11,-11],[11,-13],[12,-11],[9,-11],
#         [-10, 10],[-10,11],[-12,13],[-12,9],[-13,13],[-14,12],
#         # [-9,9],[-9,8],[-8,9],[-11,11],[-11,13],[-12,10],[-9,11],
#         # [-10, -10],[-10,-12],[-10,-8],[-12,-13],[-13,-13],[-14,-12],
#         # [-9,-9],[-9,-8]
#     ],
#     columns=["x", "y"]
# )

# P = dataframe.values
# Plist = P.tolist()



#使用欧式距离
def distanceO(p1, p2):
    res = 0.0
    # Pd = []
    # # if(p1!=p2):
    # Pd.append(p1)
    # Pd.append(p2)
    for h in range(len(p1)): 
        res += np.linalg.norm(p1[h]-p2[h])
    return res

def embedding_distance(feature_1, feature_2):
    dist = np.linalg.norm(feature_1 - feature_2)
    return dist

def distanceJ(p1, p2):
    res = 0.0
    Pd = []
    Pd.append(p1)
    Pd.append(p2)
    res = dist.pdist(Pd,'jaccard') 
    return res


def irred_k_means(Qlist,m,k,C,e,Sum):

# 1. If m = 0
# Assign the points in Q to the
# nearest centers in C.
# Sum = Sum + The clustering cost of Q.
# Return the clustering.
    global Cen
    global Seq
    global CCenter
    global Clu
   # global Ssum
    if m == 0:
        
        # list1 = []
        # Cen = [list(list1) for i in range(0,k+1)]
        # list1 = []
        Qcost = []
        csum = []
        # Cen = [list(list1) for i in range(0,len(C))]
       # print("C",len(C),C)
        for i in range(0,len(Qlist)):
            dis = []
            for j in range(0,len(C)):
                dis.append(embedding_distance(Qlist[i],C[j]))
            a = dis.index(min(dis))
            Qcost.append(min(dis))
            #print("a",a)
            #Cen[a].append(Qlist[i])
        csum.append(np.array(C))    
        Sum = Sum + sum(Qcost[:len(Qcost)])
        csum.append(Sum)       
        Csum[k-1].append(csum)
        #list1 = []
       # list2 = [list(list1) for i in range(k)]
        #Cen = copy.deepcopy(list2)
        Clu.append(Sum)
        #return Cen
        return Sum

    
# 2. (a) Sample a set S of size O  k α2 from Q.
# (b) For each set subset S of S of size O  1α do Compute the centroid c of S.

   
    b = 1/k
    Ssample = []
    if len(Qlist) > round(4/(b*e)):
        QQlist = random.sample(Qlist,round(4/(b*e)))   
        
        for i in range(round(2/b)):
            Ssample.append(random.sample(QQlist,round(2/e)))
        #print(1,len(Ssample))
    else:           
       # if len(Qlist) < round(2/e):
        Ssample.append(Qlist) 
        #print(2)
        # else:
        #     for i in range(math.floor(len(Qlist)/round(2/e))):
        #         Ssample.append(random.sample(Qlist,round(2/e)))
            #print(3,len(Ssample))
                
            
    for h in range(len(Ssample)):
       # h = 0
        # CCenter[h].clear()
        Cf = copy.deepcopy(C)
        Cf.append(np.mean(np.array(Ssample[h]), axis=0))
        # CCenter[h] = copy.deepcopy(Cf)
        #Csum[0] = [i for i in Csum[0] if i != ""]
       # print("Csum",len(CCenter),CCenter)
        #m=m-1
        #C[h] = copy.deepcopy(Csum[h])
    #     print(len(Qlist),m-1,k,Cf,e,Sum)
    # #调试
    #         m=m-1
    #         C=Cf
#        if Sum < min(Clu):
        irred_k_means(Qlist,m-1,k,Cf,e,Sum)
    # Csum[k-1] = copy.deepcopy(CCenter)
# 3. (a) Consider the points in Q in ascending order of distance from C.
    if m!=k:
        Seq = []       
        Dis = []#Q到最近C的距离
        for i in range(0,len(Qlist)):
            dis = []
            for j in range(0,len(C)):
                dis.append(embedding_distance(Qlist[i],C[j]))
            dis = sorted(dis)
            Dis.append(dis[0])
        seq = sorted(Dis) #Q到最近C的距离进行排序
        Diss = copy.deepcopy(Dis)
        qlist = copy.deepcopy(Qlist)
    # (b) Let U be the first |Q|/2 points in this sequence.
        for i in range(0,len(seq)):#长度|P|
            for j in range(0,len(Diss)):
                if (Diss[j] == seq[i]):
                    Seq.append(qlist[j])#按距离C最近顺序排列点，长度为|Q|
                    #print(len(Seq))
                    del Diss[j]
                    del qlist[j]
                    break
    
        U = Seq[:math.floor(len(Seq)/2)]#前二分之一的点，长度+|Q|/2      
    # (c) Assign the points in U to the nearest centers in C.
    #???    
        # list1 = []
        # Ccen = [list(list1) for i in range(6)]
        for i in range(0,len(U)):
            dis = []
            for j in range(0,len(C)):
                dis.append(embedding_distance(U[i],C[j]))
            a = dis.index(min(dis))
            Cen[a].append(U[i])#把U分配至C，形成集合
       # Cen[k-1].append(Ccen)
    # (d) Sum = Sum + The clustering cost of U.
        Ucost = sum(seq[:math.floor(len(seq)/2)])#所有距离前一半的和
        Sum = Sum + Ucost   
        Qlist = copy.deepcopy(Seq[math.floor(len(Seq)/2):] ) #Q=Q-U，长度为|Q|/2 
        if len(Qlist) < round(4/(b*e)):
            #print(3)
            C.append(np.mean(np.array(Qlist), axis=0))
            m = 0
        # print(len(Qlist),m,k,C,e,Sum)
        if Sum < min(Clu):
            irred_k_means(Qlist,m,k,C,e,Sum)

    


# (e) Compute the clustering
# Irred-k-means(Q − U, m, k, C, α, Sum).
# 4. Return the clustering which has minimum cost.
    
e = 0.9
K = 4
Clu = []
bestclu = []
list1 = []
Csum = [list(list1) for i in range(K)]
for i in range(1,K+1):
    f = [] 
    list2 = [list(list1) for i in range(i)]
    Cen = copy.deepcopy(list2)
    irred_k_means(Plist,i,i,f,e,0)
    minsum = Csum[0][0][-1]
    #输出最新集合
    if len(Csum[i-1])>1:
        
        for j in range(len(Csum[i-1])):         
            if minsum > Csum[i-1][j][-1]:
                minsum = Csum[i-1][j][-1]
                bestclu = Csum[i-1][j]
print (bestclu[-2],bestclu[-1])
C = bestclu[-2]
list1 = []
list2 = [list(list1) for i in range(K)]
Cen = copy.deepcopy(list2)
for i in range(0,len(Plist)):
            dis = []
            for j in range(0,len(C)):
                dis.append(embedding_distance(Plist[i],C[j]))
            a = dis.index(min(dis))
            Cen[a].append(P[i])
Dis = []#P到最近C的距离
for i in range(0,len(Plist)):
    dis = []
    for j in range(0,len(C)):
        dis.append(embedding_distance(Plist[i],C[j]))
    dis = sorted(dis)
    Dis.append(dis[0])
Cost = sum(Dis)

    # list1 = []
    # CCenter= [list(list1) for i in range(5)]
#     #Clu.append(Cen)
# 调试
# list2 = [list(list1) for i in range(K)]
# Cen = copy.deepcopy(list2)
# Qlist = copy.deepcopy(Plist)
# m = 4
# k = 4
# C = []
# Dis = []
# Sum = 0
# print(len(Qlist),m,k,C,a,Sum)
P_pred = []
for i in range(0,len(P)):
    for j in range(0,len(Cen)):
        for h in range(0,len(Cen[j])):
            # print(P[i],Cen[j][h])
            # print(j)
            if (P[i] == Cen[j][h] ).all():
                #print(P[i],Cen[j][h])
                #print(b)    
                P_pred.append(j)
p_pred = np.array(P_pred)
print(p_pred)

               
end1 = time.perf_counter()
print(end1-start)            

#long running
#do something other


samplesArr = P


##输入k-means结果，质心作为单纯形顶点
u = C
U = []
for i in range(len(u)):
    U.append(u[i])
#在集合U中构建单纯形，应用单纯形引理在其内部构建网格，建立索引                 
k = len(U)#单纯形点
n = 5 #网格化迭代次数
OriginalPoint = U
Layer_PointNum = np.empty(n)
Sum_PointNum = np.empty(n)
sum_pointnum = np.empty(n-1)#不含原始层的数量和
Layer_PointNum[0] = k
Sum_PointNum[0] = Layer_PointNum[0]
Layer_PointNum[1] = Layer_PointNum[0] * (Layer_PointNum[0]-1)/2
Layer_PointNum[2] = Layer_PointNum[0] * Layer_PointNum[1] + Layer_PointNum[1] * (Layer_PointNum[1]-1)/2
Layer_PointNum[3] = Layer_PointNum[0] * Layer_PointNum[2] + Layer_PointNum[1] * Layer_PointNum[2] + Layer_PointNum[2] * (Layer_PointNum[2]-1)/2
Layer_PointNum[4] = Layer_PointNum[0] * Layer_PointNum[3] + Layer_PointNum[1] * Layer_PointNum[3] + Layer_PointNum[2] * Layer_PointNum[3] + Layer_PointNum[3] * (Layer_PointNum[3]-1)/2
for i in range (1,n):
    Sum_PointNum[i] = Sum_PointNum[i-1] + Layer_PointNum[i]
    if (i < n): 
        sum_pointnum[i-1] = Sum_PointNum[i] - Layer_PointNum[0]
#nIndex = 0
# IndexList=pd.DataFrame(Index,columns={"bLayer1":"","ulIndex1":"","bLayer2":"","ulIndex2":""},index=[0])
PointIndex = []
pointindex = []


for k in range (1,n):
    LayerNO = k;
    if (LayerNO == 1):
        for i in range (0,int(Layer_PointNum[0])):      
          for j in range (i+1,int(Layer_PointNum[0])):                          
                  index = [0,i,0,j]
                  PointIndex.append(index)
                  #nIndex = nIndex + 1
    else:
        for l in range (0,n-1):  
            if (l != LayerNO-1):
                if (l < LayerNO-1):
                    for i in range (0,int(Layer_PointNum[l])):      
                      for j in range (0,int(Layer_PointNum[LayerNO-1])):                                  
                              index = [l,i,LayerNO-1,j]
                              PointIndex.append(index)
            else:
                for i in range (0,int(Layer_PointNum[LayerNO-1])):      
                          for j in range (0,int(Layer_PointNum[LayerNO-1])):                              
                              if (i < j and i != j):    
                                  index = [LayerNO-1,i,LayerNO-1,j]
                                  PointIndex.append(index) 
for i in range(0,n-1):  
    if(i == 0):
        pointindex.append(PointIndex[0:int(sum_pointnum[i])]) 
    else:
        pointindex.append(PointIndex[int(sum_pointnum[i-1]):int(sum_pointnum[i])])
        
####使用递归算法遍历所有点进行真值计算
Os = []
pc = []
pcindex = []
W = []
for i in range(1,5):
    for j in range(0,len(pointindex[i-1])): 
        bIs = True
        p = 0
       # print(i,j,len(pointindex[i-1]))
        pci = [i,j]
        pcindex.append(pci)
        p = sub.GetSegmentPoint(i,j,pointindex,OriginalPoint,p,bIs) 
        #pc.append(p)
        #论文 （2-5） 计算权重
        w = sub.weights(p, samplesArr, len(samplesArr)) 
        W.append(w)
        #计算所有点目标值OS     
        os = sub.objectFunction(p, w, samplesArr, n=len(samplesArr)) #计算目标函数
        Os.append(os)
        s = os    
        #找到目标值OS最小值对应的点，此点即为真值点，计算权重w并写入文件           

s = sorted(Os) #排序 从小到大 
for i in range(0, len(Os)):
    if (s[0] == Os[i]): #s0最小 通过比较，找到最小的os i
        pstar = sub.GetSegmentPoint(pcindex[i][0],pcindex[i][1],pointindex,OriginalPoint,p,bIs) #最小的p*转为dataframe格式
        print(pstar)
        #Result.append(re)
        # result.to_csv("trueValue"+str(count)+str(1)+".csv")#把上一步的p*写为csv
        #result.to_csv("./final_result/final50/trueValue"+str(count)+".csv")#lcb原路径
        wresult = sub.weights(pstar, samplesArr, n=len(samplesArr)) #计算权重
        weights = pd.DataFrame(w)
        # weights.to_csv("weights"+str(count)+str(1)+".csv")
        #weights.to_csv("./final_result/final50/weights"+str(count)+".csv")#lcb源路径
        break

end2 = time.perf_counter()
print(end2-start)            


# # 根据类别分割数据后，画图

# color = ['r', 'g', 'b', 'c', 'y', 'm', 'k']
# ax = plt.subplot()
# f = open('C:/Users/84046/Desktop/2021.12真知发现/结果.csv', 'w', encoding='utf-8', newline='')
# csv_writer = csv.writer(f)

# for p in range(0,len(p_pred)):
#     y=p_pred[p]
#     csv_writer.writerow([y])
#     ax.scatter(P[p, 0], P[p, 1], c=color[int(y)])
# for i in range(len(C)):
#     ax.scatter(C[i, 0], C[i, 1], c='y',marker='^',s=150)
# ax.scatter(pstar[0], pstar[1], c='m',marker='*',s=200)
# # for i in range(len(C)-1):  
# #     print(C[i], C[i+1])
# #     plt.plot(C[i], C[i+1], color='r')
# f.close()
# plt.show() 
