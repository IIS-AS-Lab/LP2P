import cv2
import tifffile
from skimage import io
import os.path
from matplotlib import pyplot as plt
import numpy as np
import sys

### import time
### from numpy import linalg as LA

argc_no = len(sys.argv)
if (argc_no < 2):
   print("No folder input")
   exit()

input_file = " "
out_file = " "
i=1
show_sp = 0
smooth_sp = 1 
no_smooth_sp = 0
sum_sp = 0
average_sp = 0

L_st = 0
L_end = 0
v_line = []

sm_no = 1    ### the number of smoothing

while (i<argc_no ):
   if (sys.argv[i] == "-i"):
      input_file = sys.argv[i+1]
      i += 2
   elif (sys.argv[i] == "-o"):
      out_file = sys.argv[i+1]
      i += 2
   elif (sys.argv[i] == "-start"):
      L_st = (int)(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-end"):
      L_end = (int)(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-sm_no"):
      sm_no = (int)(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-v_line"):
      v_line.append((int)(sys.argv[i+1]))
      i += 2
   elif (sys.argv[i] == "-show"):
      show_sp = 1
      i += 1
   elif (sys.argv[i] == "-sum"):
      sum_sp = 1
      i += 1
   elif (sys.argv[i] == "-mean"):
      average_sp = 1
      i += 1
   elif (sys.argv[i] == "-average"):
      average_sp = 1
      i += 1
   elif (sys.argv[i] == "-no_smooth"):
      smooth_sp = 0
      no_smooth_sp = 1
      i += 1
   elif (sys.argv[i] == "-nosmooth"):
      smooth_sp = 0
      no_smooth_sp = 1
      i += 1
   else:
      i += 1 
if input_file == " ":
   print("No file name input")
   exit()
if smooth_sp == 0:
   print("If no smoothing, noise peaks may be amplified and generate unexpected highlight regions")
 
print("Input file = ",input_file)

try:
   t = tifffile.TiffFile(input_file)
except IOError:
   print(f"\nOpen file \"{input_file}\" error! \nQuit")
   quit()

length = len(t.pages)
(Row, Col) = t.pages[0].shape
input_type = str(t.pages[0].dtype)

print(f"org video ({Row},{Col})  L = {length} data type = {input_type}")
if L_end == 0:
   L_end = length

print(f"output length = {L_end-L_st} from {L_st} to {L_end}")

if length <5:
   print("Length is too short. Not a video !")
   quit()

int_sp = 0
if smooth_sp == 1:
   if 'uint16' in input_type or 'int16' in input_type or 'uchar' in input_type:
      int_sp = 1
      print(f"int smoothing")

frame_max = t.pages[5].asarray() 

the_key = None
def press(event):
   global the_key
   the_key = event.key

if smooth_sp == 1 and int_sp == 1:
   frame_max = cv2.medianBlur(frame_max,5)

print(f"smooth_sp = {smooth_sp}")

frame_min = np.copy(frame_max)

if "uint" not in input_type:
   frame_max = frame_max.astype(np.int32)
   frame_min = frame_min.astype(np.int32)
"""
frame_max[:,:] = -999999
frame_min[:,:] = 999999
"""
star_no = 100
star_step = (int)((L_end-L_st)/star_no)

for i in range(star_no):
   print(".",end="",flush=True)
for i in range(star_no):
   print("\x08",end="",flush=True)

f_no = 1
P100 = []
P99 = []
P99_5 = []
P0 = []
P_STD = []
P_MEAN = []
frame_sum = t.pages[0].asarray()
frame_sum = frame_sum.astype(int)
for i in range(L_st+1,L_end-L_st-1):
   if (i % star_step == 0):
      print("@",end="",flush=True)
   frame = t.pages[i].asarray()
   if show_sp > 0:
      P100.append((int)(np.max(frame))) 
      P0.append((int)(np.min(frame))) 
      P99.append((int)(np.percentile(frame,99)))
      P_MEAN.append((int)(np.mean(frame)))
   """
   if i < 5:
      print(f"{i}  Max = {np.max(frame)}  Min = {np.min(frame)}")
   """
   if smooth_sp == 1 and int_sp == 1:
      ####frame = frame.astype(np.int16)
      frame = cv2.medianBlur(frame,5)

   if "uint" not in input_type:   
      frame = frame.astype(np.int32)

   frame_max = np.maximum(frame_max,frame)
   frame_min = np.minimum(frame_min,frame)
   frame_sum = frame_sum + frame.astype(int) 
print("")

frame_avg = (frame_sum/(L_end-L_st)).astype(np.uint16)

if show_sp > 0:
   p100 = np.array(P100)
   p99 = np.array(P99)
   p_mean = np.array(P_MEAN)
   p0 = np.array(P0)


   print(f"P100   Max={np.max(p100)} Min={np.min(p100)} ",end="")
   print(f"Avg = {np.mean(p100):.2f} Std = {np.std(p100):.2f}")

   print(f"P99   Max={np.max(p99)} Min={np.min(p99)} ",end="")
   print(f"Avg = {np.mean(p99):.2f} Std = {np.std(p99):.2f}")

   print(f"Mean   Max={np.max(p_mean)} Min= {np.min(p_mean)} ",end="")
   print(f"Avg = {np.mean(p_mean):.2f} Std = {np.std(p_mean):.2f}")

   print(f"P0   Max={np.max(p0)} Min={np.min(p0)} ",end="")
   print(f"Avg = {np.mean(p0):.2f} Std = {np.std(p0):.2f}")

   p99_min_sp = np.argmin(p99)
   pmean_min_sp = np.argmin(p_mean)
   print(f"pmean_min_sp = {pmean_min_sp}   p99_min_sp = {p99_min_sp}")

   plt.figure("P100",figsize=(8,4))
   plt.title(input_file+"  P100&P99&Mean")
   plt.plot(p100)
   plt.plot(p99)
   plt.plot(p_mean)
   plt.plot(p0)
   if len(v_line) > 0:
      for i in range(len(v_line)):
         plt.axvline(x=v_line[i],color = 'r')
   plt.waitforbuttonpress()
   plt.close()
   """
   plt.figure("P0",figsize=(8,4))
   plt.title(input_file+"  P0&Mean")
   plt.plot(p0)
   plt.plot(p_mean)
   plt.waitforbuttonpress()
   plt.close()
   """
##################################################################################
### Keyboard pause
##################################################################################
the_key = None
def press(event):
   global the_key
   the_key = event.key

if out_file == " ":
   if input_file[-5:] == ".tiff":
      input_file = input_file[:-5]
   if input_file[-4:] == ".tif":
      input_file = input_file[:-4]

min_f_min = np.min(frame_min)
if min_f_min < 0:
   frame_min = frame_min.astype(np.int16)
else:
   frame_min = frame_min.astype(np.uint16)
   if smooth_sp == 1:
      print("Min smoothing")
      for i in range(sm_no):
         frame_min = cv2.medianBlur(frame_min,5)
      
min_f_max = np.min(frame_max)
if min_f_max < 0:
   frame_max = frame_max.astype(np.int16)
else:
   frame_max = frame_max.astype(np.uint16)
   if smooth_sp == 1:
      print("Max smoothing")
      for i in range(sm_no):
         frame_max = cv2.medianBlur(frame_max,5)
F_max_min = frame_max - frame_min

F_max_min = F_max_min.astype(np.uint16)
if smooth_sp == 1:
   print("MaxMin Smoothing")
   for i in range(sm_no):
      F_max_min = cv2.medianBlur(F_max_min,5)
   input_file = input_file +".sm"
   if sm_no >1:
      input_file += str(sm_no)

print(f"({frame_max.dtype}) max of frame_max = {np.max(frame_max)}  min of frame_max ={np.min(frame_max)}")
print(f"({frame_min.dtype}) max of frame_min = {np.max(frame_min)}  min of frame_min ={np.min(frame_min)}")

print(f"({F_max_min.dtype}) max of F_max_min = {np.max(F_max_min)}  min of F_max_min ={np.min(F_max_min)}")

if show_sp == 1:
   plt.figure(input_file,figsize=(8,6))
   plt.title(input_file+" Max(F) " +str(length)+" frames")
   plt.imshow(frame_max)
   plt.waitforbuttonpress()
   plt.close()
print(f"{input_file}.max.tiff")
tifffile.imwrite(input_file+".max.tiff",frame_max,photometric='minisblack')

if show_sp == 1:
   plt.figure(input_file,figsize=(8,6))
   plt.title(input_file+" Min(F) " +str(length)+" frames")
   plt.imshow(frame_min)
   plt.waitforbuttonpress()
   plt.close()

print(f"{input_file}.min.tiff")
tifffile.imwrite(input_file+".min.tiff",frame_min,photometric='minisblack')

if show_sp == 1:
   plt.figure(input_file,figsize=(8,6))
   plt.title(input_file+" Max(F)-Min(F) " +str(length)+" frames")
   plt.imshow(F_max_min)
   plt.waitforbuttonpress()
   plt.close()

print(f"{input_file}.MaxMin.tiff")
tifffile.imwrite(input_file+".MaxMin.tiff",F_max_min,photometric='minisblack')

if sum_sp == 1:
   print(f"{input_file}.sum.tiff")
   tifffile.imwrite(input_file+".sum.tiff",frame_sum,photometric='minisblack')

if average_sp == 1:
   print(f"{input_file}.mean.tiff")
   tifffile.imwrite(input_file+".mean.tiff",frame_avg,photometric='minisblack')

exit()

