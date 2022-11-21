
import sys

import numpy as np

import myplots as plts
#import seaborn as sns

path = sys.argv[1]

pdata = {}

thresholds = [1,5,10]

#gorder = ['Identical','SoA','FMSA [t=1]','FMSA [t=5]','FMSA [t=10]']
#gorder = ['Identical','FMSA [t=1]','FMSA [t=5]','FMSA [t=10]']
gorder = ['FMSA [t=1]','FMSA [t=5]','FMSA [t=10]','SalSSA [t=1]','SalSSA [t=5]','SalSSA [t=10]']
#BlueRedPallete = [ sns.color_palette("Blues_r", n_colors=6)[4],sns.color_palette("Blues_r", n_colors=6)[2],sns.color_palette("Blues_r", n_colors=6)[0] , sns.color_palette("Reds_r", n_colors=6)[4],sns.color_palette("Reds_r", n_colors=6)[2],sns.color_palette("Reds_r", n_colors=6)[0]]
BlueRedPallete = ['#76ABCA','#045C90','#021C2C','#FB8E7F','#E32D14','#440E06']



def getAvgDiv(data1, data2):
  if isinstance(data1, (list,)):
    if len(data1)!=len(data2) and False:
      val1 = np.mean(data1)
      val2 = np.mean(data2)
      val1 = val1 if val1!=0 else 0.01
      val2 = val2 if val2!=0 else 0.01
      return (val1/val2)
    else:
      data = []
      for i in xrange(len(data1)):
        val1 = data1[i] if data1[i]!=0 else 0.01
        val2 = data2[i] if data2[i]!=0 else 0.01
        data.append( val1/val2 )
      return data
  else:
    val1 = data1
    val2 = data2
    val1 = val1 if val1!=0 else 0.01
    val2 = val2 if val2!=0 else 0.01
    return (val1/val2)

def getAvgDivPercent(data1, data2):
  if len(data1)!=len(data2):
    val1 = np.mean(data1)
    val2 = np.mean(data2)
    val1 = val1 if val1!=0 else 0.01
    val2 = val2 if val2!=0 else 0.01
    return (1-(val1/val2))*100.0
  else:
    data = []
    for i in xrange(len(data1)):
      val1 = data1[i] if data1[i]!=0 else 0.01
      val2 = data2[i] if data2[i]!=0 else 0.01
      data.append( (1-val1/val2)*100.0 )
    return data

for t in thresholds:
  filename = path+'/n'+str(t)+'/compilation.csv'
  data = {}
  with open(filename) as f:
    for line in f:
      vals = [val.strip() for val in line.strip().split(',')]
      if vals[0] not in data.keys():
        data[vals[0]] = {}
      ftype=''
      name = vals[1]
      if '.' in name:
        tmp = name.split('.')
        ftype = tmp[0]+'.'
        name = tmp[1]
      if name=='fm':
        name = 'FMSA [t='+str(t)+']'
      if name=='fm2':
        name = 'SalSSA [t='+str(t)+']'
      if name=='fmsa':
        name = 'FMSA [t='+str(t)+']'
      if name=='llvm':
        name = 'Identical'
      if name=='soa':
        name = 'SOA'
      timetmp = vals[2].split(':')
      tid = len(timetmp)-1
      factor = 1
      time = 0
      while tid>=0:
        time += float(timetmp[tid].strip())*factor
        factor *= 60
        tid -= 1
      if (ftype+name) not in data[vals[0]].keys():
        data[vals[0]][ftype+name] = []
      data[vals[0]][ftype+name].append(time)

  for k in data.keys():
    if k not in pdata.keys():
      pdata[k] = {}
    for name in gorder:
      if name in data[k].keys():
        pdata[k][name] = getAvgDiv(data[k][name],data[k]['baseline'])

print pdata

#BlueRedPallete = ['black',sns.color_palette("Blues_r", n_colors=3)[0],sns.color_palette("Reds_r", n_colors=6)[4],sns.color_palette("Reds_r", n_colors=6)[2],sns.color_palette("Reds_r", n_colors=6)[0],sns.color_palette("Greens_r", n_colors=3)[0]]
#BlueRedPallete = ['black',sns.color_palette("Reds_r", n_colors=6)[4],sns.color_palette("Reds_r", n_colors=6)[2],sns.color_palette("Reds_r", n_colors=6)[0],sns.color_palette("Greens_r", n_colors=3)[0]]

plts.bars(pdata,'Normalized Time',groups=gorder,palette = BlueRedPallete, edgecolor='black',labelAverage=True,decimals=2,filename=path+'/compilation-time.pdf')
#,legendPosition='upper right')



