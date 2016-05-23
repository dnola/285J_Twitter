import pickle
import numpy as np
import gmplot
import pandas as pd

# pip install gmplot
# use "2to3 -w *.py" to convert gmplot to python 3 (sudo it on unix-likes)
# you have to do this in the directory where the module gets installed
# on my system it goes to /Users/davidnola/anaconda/lib/python3.5/site-packages/gmplot

(W, H) = pickle.load(open('NMF_500_topics_WH.pkl','rb'))
df = pickle.load(open('pandas_data.pkl','rb'))

topics = np.argmax(W,axis=1)
df['topic']=topics

# t1 = df[df['topic']==15]
t1 = df[df['topic']==10]
t1 = t1[t1['gps_precision']==10.0]

print(t1.head())

gmap = gmplot.GoogleMapPlotter(41.344109, 2.1334387, 11)

lats = t1['latitude'].tolist()
longs = t1['longitude'].tolist()
print(len(lats))

gmap.heatmap(lats, longs,dissipating=True,radius=40)
gmap.draw("mymap.html")

print(topics[:10])