
import sys

import numpy as np

import myplots as plts

path = sys.argv[1]
data = {}

thresholds = [1,5,10]

for t in thresholds:
  filename = path+'/n'+str(t)+'/results.csv'
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
      if name=='fm3':
        name = 'Oracle [t='+str(t)+']'
      if name=='fm2':
        name = 'SalSSA [t='+str(t)+']'
      if name=='fm':
        name = 'FMSA [t='+str(t)+']'
      if name=='llvm':
        name = 'Identical'
      if name=='llfm':
        name = 'Identical'
      if name=='soa':
        name = 'SOA'
      data[vals[0]][ftype+name] = float(vals[2])

gorder = ['FMSA [t=1]','FMSA [t=5]','FMSA [t=10]','SalSSA [t=1]','SalSSA [t=5]','SalSSA [t=10]']

pdata = {}
for k in data.keys():
  if 'o.bl' not in data[k].keys():
    continue
  #if len(data[k].keys())!=max([len(data[ek].keys()) for ek in data.keys()]):
  #  print 'skipping',k
  #  continue
  pdata[k] = {}
  for name in gorder:
    val = data[k]['o.'+name]
    pdata[k][name] = (data[k]['o.bl']/val-1)*100
    print k, name, pdata[k][name]

#BlueRedPallete = [ sns.color_palette("Blues_r", n_colors=6)[4],sns.color_palette("Blues_r", n_colors=6)[2],sns.color_palette("Blues_r", n_colors=6)[0] , sns.color_palette("Reds_r", n_colors=6)[4],sns.color_palette("Reds_r", n_colors=6)[2],sns.color_palette("Reds_r", n_colors=6)[0]]
BlueRedPallete = ['#76ABCA','#045C90','#021C2C','#FB8E7F','#E32D14','#440E06']


plts.bars(pdata,'Reduction (%)',groups=gorder,palette = BlueRedPallete,edgecolor='black',labelAverage=True,decimals=1,legendPosition='upper left',filename=path+'/code-size-reduction.pdf')
