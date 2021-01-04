#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


tf.reset_default_graph()
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('Stockmodel/model_7_fixdays_15_2_buy/Stock_InceptionResnet_100.ckpt.meta')
    #saver.restore(sess, tf.train.latest_checkpoint('Stockmodel/model_4/'))
    sess.run(tf.global_variables_initializer())
#     train_vars = tf.trainable_variables()
#     all_vars   = tf.global_variables()
#     for v in all_vars:
#         if "InceptionResnet_Stock" in v.name :
#             print("%s with value" % (v.name))# , sess.run(v)))
#     for v in [n.name for n in tf.get_default_graph().as_graph_def().node]:
#         print(v)
#     for v in tf.get_default_graph().get_operations():
#         print(v)
    for v in [tensor for op in tf.get_default_graph().get_operations() for tensor in op.values()]:
        if   "weights" in v.name or "biases" in v.name: continue
        if   "Placeholder" in v.name or "Prediction" in v.name: 
            print(v)
        elif node.dtype not in [tf.float32, tf.int32]: 
            print(v)
            


# In[27]:


tf.reset_default_graph()
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('Stockmodel/model_7_fixdays_15_2_buy/Stock_InceptionResnet_100.ckpt.meta')
    #saver.restore(sess, tf.train.latest_checkpoint('Stockmodel/model_4/'))
    sess.run(tf.global_variables_initializer())
#     train_vars = tf.trainable_variables()
#     all_vars   = tf.global_variables()
#     for v in all_vars:
#         if "InceptionResnet_Stock" in v.name :
#             print("%s with value" % (v.name))# , sess.run(v)))
#     for v in [n.name for n in tf.get_default_graph().as_graph_def().node]:
#         print(v)
#     for v in tf.get_default_graph().get_operations():
#         print(v)
    for v in [tensor for op in tf.get_default_graph().get_operations() for tensor in op.values()]:
        print(v); continue
        if   "Placeholder" in v.name or "Prediction" in v.name: 
            print(v)
        elif node.dtype not in [tf.float32, tf.int32]: 
            print(v)


# In[ ]:





# In[1]:


#data
fix_days = 15
tp       = 0.03
sl       = 0.02
directT  = "B"

parameter_Order = [0,1,2,0,1,3,0,2,1,0,2,3,0,3,1,0,3,2,1,3,0,1]
parameterN      = len(parameter_Order)
valDays         = ["2018-1-1","2020-1-1"]

import numpy as np
import fixDay_preprocess_3 as preprocess

target_label    ='fix_{0}days_tp{1:d}_sl{2:d}_labels_{3}'.format(fix_days, int(tp*100), int(sl*100), directT)

preprocess.preProcess(fix_days, tp, sl)
val_x  , val_y, timeRecord   = preprocess.grabData(label=target_label, startT=valDays[0]  , endT=valDays[1], getTime=True)


# In[2]:


import tensorflow as tf
import time

model_path = "Stockmodel/model_7_fixdays_15_2_buy/Stock_InceptionResnet_100.ckpt"
detection_graph = tf.Graph()

preds = []
fmaps = []
fmapMaxs = []
mapConcats = []
f1s   = []

with tf.Session(graph=detection_graph) as sess:
    # Load the graph with the trained states
    loader = tf.train.import_meta_graph(model_path+'.meta')
    loader.restore(sess, model_path)

    # Get the tensors by their variable name
    node   = detection_graph.get_tensor_by_name("Placeholder:0")
    rangeV = detection_graph.get_tensor_by_name("Placeholder_1:0")
    ans    = detection_graph.get_tensor_by_name("Placeholder_2:0")
    
    pred   = detection_graph.get_tensor_by_name("InceptionResnet_Stock/Logits/Predictions:0")
    
    featuremap    = detection_graph.get_tensor_by_name("InceptionResnet_Stock/InceptionResnet_Stock/Conv2d_base_structure_1_1x1/Relu:0")
    featuremapMax = detection_graph.get_tensor_by_name("InceptionResnet_Stock/Logits/Max:0")
    featureconcat = detection_graph.get_tensor_by_name("InceptionResnet_Stock/Logits/concat:0")
    feature_w1    = detection_graph.get_tensor_by_name("InceptionResnet_Stock/Logits/Logits/weights:0")

    # Make predictions
    a = time.time()
    for (x1,x2), y in zip(val_x, val_y):
        
        ratio = np.argmax(y)
        pred0, fmap, fmapsMax, mapsconcat, f1 = sess.run([pred, featuremap, featuremapMax, featureconcat, feature_w1], 
                                          feed_dict={node      : x1[np.newaxis, :,parameter_Order, np.newaxis], 
                                                     rangeV    : np.float32([x2])[np.newaxis, :],
                                                     ans       : y[np.newaxis,:] })
        preds.append(pred0)
        fmaps.append(fmap)
        fmapMaxs.append(fmapsMax)
        mapConcats.append(mapsconcat)
        f1s.append(f1)
print("{0:.3f}".format(time.time()-a))
    


# In[3]:


preds = np.array(preds).reshape(-1,3)
# fmaps = np.array(fmaps)
# fmapMaxs = np.array(fmapMaxs)
mapConcats = np.array(mapConcats)
f1s = np.array(f1s)


# In[4]:


pred_ans = np.argmax(preds, axis=1)
true_ans = np.argmax(val_y, axis=1)
right_ans= np.where(np.argmax(preds, axis=1)==np.argmax(val_y, axis=1))[0]


# In[5]:


#detect model result
print("p-t")
for i in range(3):
    for i2 in range(3):
        print("{0}-{1} {2}".format(i, i2, np.sum(np.all([pred_ans==i, true_ans==i2],axis=0))))
print("\n",len(preds))


# In[6]:


# pos = np.where(np.all([np.argmax(preds, axis=1)==1, np.argmax(val_y, axis=1)==1], axis=0))[0]
# for i in pos:
#     print(i,"  len=",  val_x[i][0].shape,"timeRecord=", timeRecord[i],"pred=",preds[i])
#263


# In[7]:


import datetime
import pandas as pd

def History_Report(data, timeRecord, preds, inType=[1], fixdays=15, tp=0.03, sl=0.02, commission=0.005, limitdays=3):
    data0 = np.array(data[['open','close']].copy())
    poses = np.where(np.any([np.argmax(preds, axis=1)==t for t in inType], axis=0))[0]
    p0    = None
    for pos in poses:
        d2 = datetime.datetime(*np.int32(timeRecord[pos].strftime("%Y-%m-%d").split("-")))
        n  = np.where(data.index>=d2)[0][0]

        # B
        tp0= data0[n,0] + data0[n,0]*tp
        sl0= data0[n,0] - data0[n,0]*sl

        tpn= np.where(data0[n+limitdays-1:n+fixdays,1]>=tp0)[0]
        sln= np.where(data0[n+limitdays-1:n+fixdays,1]<=sl0)[0]
        tpn= tpn[0]+n+limitdays-1 if len(tpn)!=0 else n+fixdays
        sln= sln[0]+n+limitdays-1 if len(sln)!=0 else n+fixdays

        fin= min(min(min(tpn,sln), n+fixdays-1),len(data0)-1)
        p1 = pd.DataFrame({"InDate"         : [data.index[n].to_pydatetime().strftime("%Y-%m-%d")],
                           "OutDate"        : [data.index[fin].to_pydatetime().strftime("%Y-%m-%d")],
                           "LotType"        : ["B"],
                           "InPrice"        : [data0[n,0]],
                           "OutPrice"       : [data0[fin,1]],
                           "Profit"         : [data0[fin,1]-data0[n,0]],
                           "Net Profit"     : [data0[fin,1]-data0[n,0]*(1+commission)],
                           "Profit rate"    : [(data0[fin,1]-data0[n,0])/data0[n,0]],
                           "Net Profit rate": [(data0[fin,1]-data0[n,0])/data0[n,0]-commission],
                           "Commission"     : [commission],
                           "InPos"          : [n],                      
                           "OutPos"         : [fin]
                          })
        if p0 is None: 
            p0 = p1
        else :
            p0 = p0.append(p1) 
    return p0

LotsReport = History_Report(preprocess.k_plot, timeRecord, preds, inType=[1], fixdays=15, tp=0.03, sl=0.02, commission=0.005, limitdays=3)
# LotsReport.to_csv("Lots_Report.csv", index=False)


# In[8]:


import numpy as np
from scipy import interpolate
import pylab as pl
import matplotlib.pyplot as plt

path = "Report/model_7_fixdays_15_2_buy/Total/record"
featureN = 0 #"N:0/ B:1/ S:2 "
colors   = ["mediumblue", "r", 'g']

for label_i in range(len(val_y)):
    featureN= np.argmax(preds[label_i])
    record  = np.zeros([fmaps[label_i].shape[1]], dtype=np.float32)

    a = f1s[label_i][:,featureN]
    b = mapConcats[label_i].reshape([-1])
    c = (a*b)

    for i,v in enumerate(c[:1024]):
        pos = np.argmax(fmapMaxs[label_i][0,i,:])
        record[pos] = record[pos]+v

    for i,v0 in enumerate(c[2048:-1].reshape([-1,5])):
        for i2, v in enumerate(v0):
            record[i2-5] = record[i2-5]+v
        
    disLook = c[-1]
    
    shape = val_x[label_i][0].shape[0]
    
    n   =len(record)
    mul =val_x[label_i][0].shape[0]/n 
    x   =np.linspace(0, n*mul-1, n)
    y   =record
    xnew=np.linspace(0,n*mul-1,shape)
    
    f=interpolate.interp1d(x,y,kind="quadratic")
    ynew=f(xnew)
    
    d2 = np.int32(timeRecord[label_i].strftime("%Y-%m-%d").split("-"))
    
    preprocess.K_plot(preprocess.k_plot, d1=shape, d2=d2, Indicator=ynew, Main=None, name="{0}_{1}.jpg".format(path, label_i), color=colors[featureN], report=LotsReport)
    plt.close('all')


# In[4]:


import matplotlib.pyplot as plt
import numpy as np
import cv2

def write(img, w, loc, color, size=1):
    cv2.putText(img, w, loc, cv2.FONT_HERSHEY_COMPLEX, size, color, 2)
    
def recordTable(img, Orders, Simple, Compound, Commission=0.5):
    colors = [0,0,0,0]
    for i in  range(2):
        colors[i*2]   = (0,128,255)  if Simple[i]  >=0 else (127, 255, 0)
        colors[i*2+1] = (0,128,255)  if Compound[i]>=0 else (127, 255, 0)
        Simple[i]   = "+"+str(round(Simple[i],2))   if Simple[i]  >=0 else str(round(Simple[i],2))
        Compound[i] = "+"+str(round(Compound[i],2)) if Compound[i]>=0 else str(round(Compound[i],2))
        
    write(img, w="{0:<12}".format('Commission:'), loc=(1070,200), color=(0,0,0))
    write(img, w="{0:>8}%".format(Commission), loc=(1320,200), color=(0,0,0))

    write(img, w="{0:<12}".format('Orders:'), loc=(1070,250), color=(0,0,0))
    write(img, w="{0:>8} ".format(Orders), loc=(1320,250), color=(0,0,0))

    write(img, w="{0:<12} ".format('Simple Interest:'), loc=(1070,300), color=(0,0,0))
    write(img, w=" - {0:<9}".format('Balance:'), loc=(1070,350), color=(0,0,0), size=0.8)
    write(img, w="{0:>8}%".format(Simple[0]), loc=(1320,350), color=colors[0])
    write(img, w=" - {0:<9}".format('Profit/Loss:'), loc=(1070,400), color=(0,0,0), size=0.8)
    write(img, w="{0:>8}%".format(Simple[1]), loc=(1320,400), color=colors[2])

    write(img, w="{0:<12} ".format('Compound Interest:'), loc=(1070,450), color=(0,0,0))
    write(img, w=" - {0:<9}".format('Balance:'), loc=(1070,500), color=(0,0,0), size=0.8)
    write(img, w="{0:>8}%".format(Compound[0]), loc=(1320,500), color=colors[1])
    write(img, w=" - {0:<9}".format('Profit/Los:'), loc=(1070,550), color=(0,0,0), size=0.8)
    write(img, w="{0:>8}%".format(Compound[1]), loc=(1320,550), color=colors[3])
    
def RecordData(data, timeRecord, report, inPath, outPath, Commission=0.005):
    Acount_S = {}
    Acount_C = {}
    Balance_S, Balance_C = 1, 1
    
    data0 = np.array(data[['open','close']].copy())
    data1 = np.array(report[['InPos', 'OutPos']].copy())
    d2 = datetime.datetime(*np.int32(timeRecord[0].strftime("%Y-%m-%d").split("-")))
    n  = np.where(data.index>=d2)[0][0]
    for i in range(len(timeRecord)):
        #Out
        pos = np.where(data1[:,1]==n+i)[0]
        if len(pos)>0:
            for pi in range(len(pos)):
                Balance_S += ((data0[n+i,1]-Acount_S[data1[pos[pi],0]][1])/Acount_S[data1[pos[pi],0]][1]-Commission)*Acount_S[data1[pos[pi],0]][0] 
                Balance_C += ((data0[n+i,1]-Acount_C[data1[pos[pi],0]][1])/Acount_C[data1[pos[pi],0]][1]-Commission)*Acount_C[data1[pos[pi],0]][0]
                del Acount_S[data1[pos[pi],0]]
                del Acount_C[data1[pos[pi],0]]
        
        #In
        pos = np.where(data1[:,0]==n+i)[0]
        if len(pos)>0:
            Acount_S[data1[pos[0],0]]=[        1,data0[n+i,0]]
            Acount_C[data1[pos[0],0]]=[Balance_C,data0[n+i,0]]
        
        #Insitu
        Profit_S, Profit_C = 0, 0
        for v in Acount_S.values():
            Profit_S += ((data0[n+i,1]-v[1])/v[1]-Commission)*v[0]
        for v in Acount_C.values():
            Profit_C += ((data0[n+i,1]-v[1])/v[1]-Commission)*v[0]
        
        #draw
        img = cv2.imread(inPath.format(i),1)

        img2 = np.ones([720,1520,3])*255
        img2[:,:1224-80,:] = img[:,80:,:]
        
        cv2.rectangle(img2, (1050,150), (1510,580), (50,50,50), 1)
        recordTable(img2, Orders=len(Acount_S), Simple=[(Balance_S-1)*100, Profit_S], Compound=[(Balance_C-1)*100, Profit_C], Commission=Commission*100)

        cv2.imwrite(outPath.format(i), img2)
       


# In[ ]:


RecordData(preprocess.k_plot, 
           timeRecord = timeRecord, 
           report     = LotsReport, 
           inPath     = "Report/model_7_fixdays_15_2_buy/Total/record_{0}.jpg", 
           outPath    = "Report/model_7_fixdays_15_2_buy/Record/record_{0}.jpg")


# In[15]:


import cv2
end_n = 501

img = cv2.imread("Report/model_7_fixdays_15_2_buy/Record/record_{0}.jpg".format(end_n), 1)

center = [525, 365]
bias   = [275, 125]
expandN= 5

for i in range(expandN+1):
    cv2.rectangle(img, (center[0]-int(bias[0]/expandN*i), center[1]-int(bias[1]/expandN*i)), (center[0]+int(bias[0]/expandN*i), center[1]+int(bias[1]/expandN*i)), (50,50,50), 1)
    img[center[1]-int(bias[1]/expandN*i)+1:center[1]+int(bias[1]/expandN*i), center[0]-int(bias[0]/expandN*i)+1:center[0]+int(bias[0]/expandN*i), :] = 255
    end_n=end_n+1; cv2.imwrite("Report/model_7_fixdays_15_2_buy/Record/record_{0}.jpg".format(end_n), img)

write(img, w="2018/01/01 - 2019/12/31 ", loc=(300, 300), color=(0, 0, 0), size=1)
end_n=end_n+1; cv2.imwrite("Report/model_7_fixdays_15_2_buy/Record/record_{0}.jpg".format(end_n), img)

write(img, w="Take Profit Lots  :  42", loc=(300, 350), color=(0, 0, 0), size=1)
write(img, w="Stop Loss Lots    :  11", loc=(300, 400), color=(0, 0, 0), size=1)
write(img, w="Time Expire Lots :   2", loc=(300, 450), color=(0, 0, 0), size=1) #(207, 174, 78)
end_n=end_n+1; cv2.imwrite("Report/model_7_fixdays_15_2_buy/Record/record_{0}.jpg".format(end_n), img)


# # Create Video

# In[17]:


import cv2
from cv2 import VideoWriter,VideoWriter_fourcc,imread,resize
import glob, os
img_root="Report/model_7_fixdays_15_2_buy/Record/record_{0}.jpg"
shape   = cv2.imread(img_root.format(0),1).shape[:2][::-1]

#Edit each frame's appearing time!
fps=5
fourcc=VideoWriter_fourcc('m', 'p', '4', 'v')
videoWriter=cv2.VideoWriter("Test-Back_without_Audio.mp4",fourcc,fps,shape)

n = len(glob.glob(os.sep.join(img_root.split(os.sep)[:-1] + ["*.jpg"])))
for i in range(n):
    imgpath = img_root.format(i)
    frame=cv2.imread(imgpath, 1)
    videoWriter.write(frame)
	
videoWriter.release()


# In[29]:


import ffmpeg

audioPath = "Report/Brand_X_Music-Buccaneer_Island.mp3"
videoPath = "Test-Back_without_Audio.mp4"
output    = "Report/Test-Back_2.mp4"

video = ffmpeg.input(videoPath)
audio = ffmpeg.input(audioPath)

ffmpeg.concat(video, audio, v=1, a=1).output(output).run(overwrite_output=True)


# In[ ]:


import ffmpeg


input_video = ffmpeg.input("../resources/video_with_audio.mp4")
added_audio = ffmpeg.input("../resources/dance_beat.ogg").audio.filter('adelay', "1500|1500")

merged_audio = ffmpeg.filter([input_video.audio, added_audio], 'amix')

(
    ffmpeg
    .concat(input_video, merged_audio, v=1, a=1)
    .output("mix_delayed_audio.mp4")
    .run(overwrite_output=True)
)


# In[25]:


import moviepy.editor as mp



audio = mp.AudioFileClip(audioPath)
video = mp.VideoFileClip(videoPath)
final = mp.concatenate_videoclips([video])
# video = video.set_duration(audio)
final = final.set_audio(audio.set_duration(final))
# video1.set_audio(audio)
#video2 = mp.VideoFileClip("video2.mp4")
# final = mp.concatenate_videoclips([video1, video2]).set_audio(audio)
# final.write_videofile("output.mp4")
final.write_videofile("Report/Test-Back_2.mp4")


# In[18]:


from moviepy.editor import VideoFileClip, AudioFileClip
 
clip_video = VideoFileClip("Test-Back_without_Audio.mp4")
clip_audio = AudioFileClip('Report/Brand_X_Music-Buccaneer_Island.mp3')

new_video = clip_video.set_audio(clip_audio)
new_video.write_videofile("Report/Test-Back.mp4")


# In[ ]:





# In[ ]:





# In[61]:


# K_plot(preprocess.k_plot, d1=(2019, 1, 1), d2=(2020, 1, 1), Indicator=None, Main=None, name='', color='mediumblue', report=LotsReport)


# In[19]:


import numpy as np
from scipy import interpolate
import pylab as pl

n   =len(record)
mul =val_x[label_i][0].shape[0]/n 
x   =np.linspace(0, n*mul-1, n)
y   =record
xnew=np.linspace(0,n*mul-1,shape)
#pl.plot(x,y,"ro")

for kind in ["quadratic"]:#插值方式 #["nearest","zero","slinear","quadratic","cubic"]
    #"nearest","zero"为阶梯插值
    #slinear 线性插值
    #"quadratic","cubic" 为2阶、3阶B样条曲线插值
    f=interpolate.interp1d(x,y,kind=kind)
    # ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of first, second or third order)
    ynew=f(xnew)
    pl.plot(xnew,ynew,label=str(kind))
pl.legend(loc="lower right")
pl.show()


# In[20]:


# preprocess.K_plot(preprocess.k_plot, d1=229, d2=d2, Indicator=ynew, Main=None, name='')


# In[9]:




img = cv2.imread("Report/model_7_fixdays_15_2_buy/Total/record_8.jpg",1)


# In[24]:


import matplotlib.pyplot as plt
import numpy as np
import cv2

def write(img, w, loc, color, size=1):
    cv2.putText(img, w, loc, cv2.FONT_HERSHEY_COMPLEX, size, color, 2)
    
def recordTable(img, Orders, Simple, Compound, Commission=0.5):
    colors = [0,0,0,0]
    for i in  range(2):
        colors[i*2]   = (0,128,255)  if Simple[i]  >=0 else (127, 255, 0)
        colors[i*2+1] = (0,128,255)  if Compound[i]>=0 else (127, 255, 0)
        Simple[i]   = "+"+str(round(Simple[i],2))   if Simple[i]  >=0 else str(round(Simple[i],2))
        Compound[i] = "+"+str(round(Compound[i],2)) if Compound[i]>=0 else str(round(Compound[i],2))
        
    write(img, w="{0:<12}".format('Commission:'), loc=(1070,200), color=(0,0,0))
    write(img, w="{0:>8}%".format(Commission), loc=(1320,200), color=(0,0,0))

    write(img, w="{0:<12}".format('Orders:'), loc=(1070,250), color=(0,0,0))
    write(img, w="{0:>8} ".format(Orders), loc=(1320,250), color=(0,0,0))

    write(img, w="{0:<12} ".format('Simple Interest:'), loc=(1070,300), color=(0,0,0))
    write(img, w=" - {0:<9}".format('Balance:'), loc=(1070,350), color=(0,0,0), size=0.8)
    write(img, w="{0:>8}%".format(Simple[0]), loc=(1320,350), color=colors[0])
    write(img, w=" - {0:<9}".format('Profit/Loss:'), loc=(1070,400), color=(0,0,0), size=0.8)
    write(img, w="{0:>8}%".format(Simple[1]), loc=(1320,400), color=colors[2])

    write(img, w="{0:<12} ".format('Compound Interest:'), loc=(1070,450), color=(0,0,0))
    write(img, w=" - {0:<9}".format('Balance:'), loc=(1070,500), color=(0,0,0), size=0.8)
    write(img, w="{0:>8}%".format(Compound[0]), loc=(1320,500), color=colors[1])
    write(img, w=" - {0:<9}".format('Profit/Los:'), loc=(1070,550), color=(0,0,0), size=0.8)
    write(img, w="{0:>8}%".format(Compound[1]), loc=(1320,550), color=colors[3])
    
def RecordData(data, timeRecord, report, inPath, outPath, Commission=0.005):
    Acount_S = {}
    Acount_C = {}
    Balance_S, Balance_C = 1, 1
    
    data0 = np.array(data[['open','close']].copy())
    data1 = np.array(report[['InPos', 'OutPos']].copy())
    d2 = datetime.datetime(*np.int32(timeRecord[0].strftime("%Y-%m-%d").split("-")))
    n  = np.where(data.index>=d2)[0][0]
    for i in range(len(timeRecord)):
        #Out
        pos = np.where(data1[:,1]==n+i)[0]
        if len(pos)>0:
            for pi in range(len(pos)):
                Balance_S += ((data0[n+i,1]-Acount_S[data1[pos[pi],0]][1])/Acount_S[data1[pos[pi],0]][1]-Commission)*Acount_S[data1[pos[pi],0]][0] 
                Balance_C += ((data0[n+i,1]-Acount_C[data1[pos[pi],0]][1])/Acount_C[data1[pos[pi],0]][1]-Commission)*Acount_C[data1[pos[pi],0]][0]
                del Acount_S[data1[pos[pi],0]]
                del Acount_C[data1[pos[pi],0]]
        
        #In
        pos = np.where(data1[:,0]==n+i)[0]
        if len(pos)>0:
            Acount_S[data1[pos[0],0]]=[        1,data0[n+i,0]]
            Acount_C[data1[pos[0],0]]=[Balance_C,data0[n+i,0]]
        
        #Insitu
        Profit_S, Profit_C = 0, 0
        for v in Acount_S.values():
            Profit_S += ((data0[n+i,1]-v[1])/v[1]-Commission)*v[0]
        for v in Acount_C.values():
            Profit_C += ((data0[n+i,1]-v[1])/v[1]-Commission)*v[0]
        
        #draw
        img = cv2.imread(inPath.format(i),1)

        img2 = np.ones([720,1520,3])*255
        img2[:,:1224-80,:] = img[:,80:,:]
        
        cv2.rectangle(img2, (1050,150), (1510,580), (50,50,50), 1)
        recordTable(img2, Orders=len(Acount_S), Simple=[(Balance_S-1)*100, Profit_S], Compound=[(Balance_C-1)*100, Profit_C], Commission=Commission*100)

        cv2.imwrite(outPath.format(i), img2)
       


# In[25]:


RecordData(preprocess.k_plot, 
           timeRecord = timeRecord, 
           report     = LotsReport, 
           inPath     = "Report/model_7_fixdays_15_2_buy/Total/record_{0}.jpg", 
           outPath    = "Report/model_7_fixdays_15_2_buy/Record/record_{0}.jpg")


# In[27]:


import glob
a = glob.glob('Report/model_7_fixdays_15_2_buy/Total/*.jpg')
sorted(a)[:20]


# In[ ]:



import cv2
import numpy as np
import glob
 
img_array = []
for filename in glob.glob('Report/model_7_fixdays_15_2_buy/Total/*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()


# In[15]:


LotsReport


# In[59]:


pred0


# In[69]:


f2


# In[32]:


f2


# In[33]:


f1


# In[35]:


fmaps.shape


# In[49]:


for i,v in enumerate(fmapsMax[0,:,:]):
    print(v)
    if i>=10: break


# In[51]:


mapsconcats[0, :10]


# In[54]:


mapsconcats[0, 1024:1034]


# In[55]:


mapsconcats[0, 2048:2058]


# In[56]:


f1.shape


# In[ ]:




