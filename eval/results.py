
import sys
import os

import string
def is_hex(s):
     hex_digits = set(string.hexdigits)
     # if s is long, then it is faster to check against a set
     return all(c in hex_digits for c in s)

def binarySize(filename):
  count = 0
  with open(filename) as f:
    for line in f:
      if ':' in line:
        tmp = line.split(':')
        offset = tmp[0].strip()
        if not is_hex(offset):
          continue
        instruction = tmp[1].strip().split('\t')
        if len(instruction) == 0:
          continue
        binary = instruction[0].strip().split()
        valid = all([is_hex(byte.strip()) for byte in binary])
        if valid:
          count += len(binary)
  return count

bench = sys.argv[1]
path = bench+'/build'

ftypes = ['bl','fm','fm2']
#fexts = ['o.','']
fexts = ['o.']


for ftype in ftypes:
  for fext in fexts:
    filename = path+'/main.'+fext+ftype
    statinfo = os.stat(filename)
    print str(bench)+', '+str(fext)+str(ftype)+', '+str(statinfo.st_size)
 # print str(bench)+', '+'txt.'+str(ftype)+', '+str( binarySize(path+'/main.txt.'+ftype) )

