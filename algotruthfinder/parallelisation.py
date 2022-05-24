# import findspark
# findspark.init('/Users/admin/spark-3.2.0')
import pyspark
from pyspark import SparkContext,SparkConf
import pandas as pd
import numpy as np
import math as m
global data_T
global read_rdd


conf = SparkConf().setAll([('spark.driver.host','localhost'), ('spark.executor.cores', '8'), ('spark.cores.max', '16'), ('spark.driver.memory','16g')])
sc = SparkContext(conf=conf)
sc

data=pd.read_csv('DS3/data.csv')
data_T=pd.DataFrame(data, columns= ['Object','Property','Value','Source'])
data_T
# constitution de dataframe pour le travail
data_T['TSInitial']=0.8
data_T['C_v']=0.0
data_T.head(5)
# Convertissons notre dataFrame en RDD
data_T.to_csv('data_T.csv',index=False)
read_ = sc.textFile("data_T.csv")
# Spliter selon les lignes de notre dataFrame
read_rdd=read_.map(lambda line: line.split(",")).map(lambda line:(line[0],line[1],line[2],line[3],line[4],line[5])).filter(lambda x: x[0]!='Object')
# Lecture de notre RDD
read_rdd.take(5)
# Definition de la matice de similarite
def similarity(dataframe):
    dict_sim = {}
    for key, df in dataframe.groupby(by=['Object','Property']):
        Values = df['Value'].unique()
        row = key[0]+key[1]
        for  i in range(len(Values)):
            w1 = Values[i]
            for  u in range(len(Values)):
                w2 = Values[u]
                sim = 1
                if w1!=w2:
                    t = abs(w1-w2)
                    sim = 1/t
                dict_sim[row+str(w1)+str(w2)] = sim
                dict_sim[row+str(w2)+str(w1)] = sim
    return dict_sim
# Declaration des variables globales

data_T=pd.read_csv('data_T.csv')
global rho
rho=0.7
global lam
lam=0.5
global dict_sim

dict_sim=similarity(data)
# la fonction Map
# (input x est un quatriplet constitutue de (object,property,valeur, TSinitial) et le renvoie 
# la cle(object,property,valeur) et la valeur sigma_v)
def MapFunction(x):
    return ((x[0],x[1],x[2]), (-m.log(1-float(x[4])), [(x[3],x[4])] ) )
def ReduceFunction(x,y):
    return x[0]+y[0],x[1]+y[1]
def Mapelement(x):
    return (x[0][0],x[0][1]),[(x[0][2],x[1][0],x[1][1])]
def ReduceShuffling(x,y):
    return x+y
def reduceFonction(x):
    out=(x[0],[])
    for v in x[1]:
        c_v=v[1]
        l=0
        for v_ in x[1]:
            if v[0]!=v_[0]:
                Key_sim=x[0][0]+x[0][1]+str(v[0])+str(v_[0])
                c_v+=rho*(v_[1]*dict_sim[Key_sim])

        CV=1/(1+np.exp(-lam*c_v))

        out[1].append((v[0],CV,v[2]))

    return out
# rdd_cv = sc.parallelize(transf.map(reduceFonction).map(mapT).reduceByKey(lambda x,y: x+y ).collect()[0][1])
def mapT(x):
    out=[]
    for v in x[1]:
        for s in v[2]:
            out.append((x[0][0],x[0][1],v[0],s[0],s[1],v[1]))
    return (1,out)
def mapTS(x):
    return x[3],x[5]
def mapReconstruit(x):
    ts = 0
    for s in source_ts:
        if x[3]==s[0]:
            ts = s[1]
            break
    return x[0],x[1],x[2],x[3],ts,x[5]
def mapTS_(x):
    return x[3],x[4]


def cos_sim(a,b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
import time as t
def truthF(data_T):
    read_ = sc.textFile("data_T.csv")
    # Spliter selon les lignes de notre dataFrame
    read_rdd=read_.map(lambda line: line.split(",")).map(lambda line:(line[0],line[1],line[2],line[3],line[4],line[5])).filter(lambda x: x[0]!='Object')
    ni=5
    trustold = 0.00001
    global source_ts 
    source_ts = []
    start = t.time()
    nb_iter_do = 0
    for i in range(ni):
        
        ts1 = list(read_rdd.map(mapTS_).reduceByKey(lambda x,y: x).sortBy(lambda x: x[0]).map(lambda x: float(x[1])).collect())
        
        fonc=read_rdd.map(MapFunction).reduceByKey(ReduceFunction)
        foncajust=fonc.map(Mapelement).reduceByKey(ReduceShuffling).map(reduceFonction)
        
        transf=sc.parallelize(foncajust.map(mapT).reduceByKey(lambda x,y: x+y ).collect()[0][1])
        
        source_ts = list(transf.map(mapTS).groupByKey().mapValues(lambda x: sum(x) / len(x)).collect())
        
        read_rdd = transf.map(mapReconstruit)
        
        ts2 = list(read_rdd.map(mapTS_).reduceByKey(lambda x,y: x).sortBy(lambda x: x[0]).map(lambda x: float(x[1])).collect())
        
        print('Itération : ',i+1)
        print(1-abs(cos_sim(ts1,ts2)))
        nb_iter_do = i+1
        if 1-abs(cos_sim(ts1,ts2)) < trustold:
            break
    end = t.time()
    return read_rdd,end-start,nb_iter_do
out = truthF(data_T)
print(out)
print(out[0].take(2))
def mapGetResult(x):
    return ((x[0],x[1]),(x[2],x[5]))
def reduceGetResult(x,y):
    return y
def reduceVote(x,y):
    if x[1] > y[1]:
        return x
    return y
out[0].map(mapGetResult)
list_A =out[0].map(mapGetResult).reduceByKey(reduceVote).map(lambda x: (x[0][0],x[0][1],x[1][0])).collect()
list_B = read_rdd.collect()
len(list_B) 
data_truth=pd.read_csv('DS3/data_truth.csv')
data_truth_=pd.DataFrame(data_truth, columns= ['Object','Property','Value'])
data_truth_.to_csv('data_truth_.csv',index=False)
read_truth = sc.textFile("data_truth_.csv")
truth_rdd=read_truth.map(lambda line: line.split(",")).map(lambda line:(line[0],line[1],line[2])).filter(lambda x: x[0]!='Object') 
list_truth=truth_rdd.collect()
len(list_truth) 
def function(list_A,list_B):
    binary=[]
    search_list = [tuple(li[:-3]) for li in list_B]
    for ser_item in search_list:
        if ser_item in list_A:
            binary.append(1)
        else:
            binary.append(0)
    return binary
        
res = function(list_A, list_B)
res_truth=function(list_truth, list_B)
# res
# res_truth 
# !pip install scikit-learn
import sklearn
from sklearn.metrics import precision_score,accuracy_score,recall_score,f1_score
print('precision',precision_score(res_truth,res))
print('accu',accuracy_score(res_truth,res))
print('recall',recall_score(res_truth,res))
print('f1',f1_score(res_truth,res))
    

