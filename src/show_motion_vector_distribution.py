import os
import cv2
from matplotlib import pyplot as plt
import time
import numpy as np
import sys
from skimage import io
import scipy.ndimage
from scipy import signal

############## Keyboard event detection ############
pause_flag=0
##########################################################   
argc_no = len(sys.argv)
if (argc_no < 2):
   print("Show motion vector distribution")
   print("No input!");
   print("-mv_file    input motion vector file")
   print("-save       save to a PNG image")
   print("-title_name set a specific title name")
   print("-range      set a fixed data point range")
   print("-color      select a colormap")
   print("-dpi        DPI of output image")
   exit()
################# Set parameters ################

smoothing_sp = 0
show_sp = 0

max_sp = 0
mean_sp = 1
pdf_sp = 0
##################################################################################
fixed_range_sp = 0
def_range = 20
##################################################################################
### Check the input parameters from argv
##################################################################################

i=1

mv_file = " "
title_name = " "
color_sp = 0
save_sp = 0
output_file = "plt.png"
dpi = 100

while (i<argc_no ):
   if (sys.argv[i] == "-mv_file"):
      mv_file = sys.argv[i+1]
      i += 2
   elif (sys.argv[i] == "-range"):
      fixed_range_sp = 1
      def_range = (int)(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-color"):
      color_sp = (int)(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-title_name"):
      title_name = sys.argv[i+1]
      i += 2
   elif (sys.argv[i] == "-output"):
      output_file = sys.argv[i+1]
      save_sp = 1
      i += 2
   elif (sys.argv[i] == "-save"):
      save_sp = 1
      i += 1
   elif (sys.argv[i] == "-dpi"):
      save_sp = 1
      pdf_sp = 1
      dpi = (int)(sys.argv[i+1])
      i += 2
   else:
      i += 1
 
######## Check input parameters ###########
if (mv_file == " "):
   print("No input motion vector file")
   exit()
print(f"Title name = {title_name}")
############ Detect Keyboard Event ###############
the_key = None
def press(event):
    global the_key
    the_key = event.key

###########################################################################
### Avg_Vect: The final shift vectors in an array format
###########################################################################

length =0
with open(mv_file) as f:
   for line in f:
      if (line[0] != '#') and (line[0] != '*'):
          length += 1
f.close()
print(f"Length = ",length)

Avg_Vect = np.zeros((length,2),dtype=int)
trgP = np.zeros((length,4),dtype=int)
imgP = np.zeros((length,4),dtype=int)

avg_list = []

i = 0
with open(mv_file) as f:
   for line in f:
      if (line[0] != '#') and (line[0] != '*'):
         fary = line.rstrip().split(',')
         if len(fary) == 10:
            avg_list.append(((int)(fary[0]),(int)(fary[1])))
            Avg_Vect[i,0] = (int)(fary[0]); Avg_Vect[i,1] = (int)(fary[1])   
            for j in range(4):
               trgP[i,j]=fary[j+2]
               imgP[i,j]=fary[j+6]
            i += 1
         else:
            print("Some data points are lost!")

if fixed_range_sp == 1:
   Rmax =    def_range; Cmax = def_range
   Rmin = -1*def_range; Cmin = -1*def_range
else: 
   Rmax = ((int)(np.max(Avg_Vect[:,0])/10)+1)*10
   Cmax = ((int)(np.max(Avg_Vect[:,1])/10)+1)*10
   if abs(Rmax) < abs(Cmax):
      Rmax = Cmax
   else:
      Cmax = Rmax

   Rmin = ((int)(np.min(Avg_Vect[:,0])/10)-1)*10
   Cmin = ((int)(np.min(Avg_Vect[:,1])/10)-1)*10

   if abs(Rmin) < abs(Cmin):
      Rmin = Cmin
   else:
      Cmin = Rmin

   if Rmin < 0:
      Rmin = -1*Rmax
      Cmin = Rmin

mag = []
#for i in range(len(Avg_Vect)):
#   mg = np.sqrt(Avg_Vect[:, 0]*Avg_Vect[:,0] + Avg_Vect[:,1]*Avg_Vect[:,1])
#   mag.append(mg)
mag = np.sqrt(Avg_Vect[:,0]*Avg_Vect[:,0] + Avg_Vect[:,1]*Avg_Vect[:,1])
a_mag = np.asarray(mag)

print("Average motion mag = ", np.mean(a_mag))
print(f"Maxmum motion mag = {np.max(a_mag)} in {np.argmax(a_mag)}")
print(f"Max positive mv at R = {np.max(Avg_Vect[:,0])} in {np.argmax(Avg_Vect[:,0])}")
print(f"Max positive mv at C = {np.max(Avg_Vect[:,1])} in {np.argmax(Avg_Vect[:,1])}")
print(f"Max negative mv at R = {np.min(Avg_Vect[:,0])} in {np.argmin(Avg_Vect[:,0])}")
print(f"Max negative mv at C = {np.min(Avg_Vect[:,1])} in {np.argmin(Avg_Vect[:,1])}")

avg_set = list(set(avg_list))
avg_set_count = []

point_count = np.zeros((len(avg_set),3),dtype=int)

for i in range(len(avg_set)):
   (point_count[i,0],point_count[i,1]) = avg_set[i] 
   point_count[i,2] = avg_list.count(avg_set[i])
   avg_set_count.append(avg_list.count(avg_set[i]))

print(f"original total number = {len(avg_list)}")
print(f"The number with duplicates = {len(avg_set)}")
print(f"Max. occurence = {max(point_count[:,2])}")

if title_name == " ":
   s_str = mv_file.replace("f.MC","f\nMC")
   s_str = s_str.replace("tiff.sm","tiff\nsm")
   s_str = s_str.replace("tiff.ns","tiff\nns")
   s_str = s_str.replace("f16.tiff.","f16.tiff\n")
else:
   s_str = title_name

print(f"Title name = {s_str}")

p_max = np.max(Avg_Vect)
p_min = np.min(Avg_Vect)

##fig = plt.figure("Correlated Motion Vectors",figsize=(7.5,6),dpi=dpi)
fig = plt.figure("Correlated Motion Vectors",dpi=dpi)

plt.axis([Cmin,Cmax,Rmin,Rmax])

### cmap='viridis'   cmap='hot'   cmap='jet'

if color_sp == 0: 
   color_mp = 'viridis'
elif color_sp == 1:
   color_mp = 'winter'
else:
   color_mp = 'jet'

plt.scatter(point_count[:,1],point_count[:,0],c=point_count[:,2],cmap=color_mp)
plt.colorbar()
plt.axvline(x = 0, linestyle = 'dashed', lw=1, color='k')
plt.axhline(y = 0, linestyle = 'dashed', lw=1, color='k')
plt.xlabel("$\\Delta$C",fontsize=18)
plt.ylabel("$\\Delta$R",fontsize=18)

if fixed_range_sp == 1:
   plt.xlim([-1*def_range,def_range])
   plt.ylim([def_range,-1*def_range])
else:
   if max(abs(p_max),abs(p_min)) < 10:
      plt.xlim([-10,10])
      plt.ylim([-10,10])

plt.tight_layout()

if save_sp == 1:
   ###plt.title("", wrap=True)
   print("Write the plot to png")
   output_file_png = mv_file + "."+color_mp+".png"
   output_file_pdf = mv_file + "."+color_mp+".pdf"
   output_file_tiff = mv_file + "."+color_mp+".tiff"
   if pdf_sp == 0:
      print(f"output_file_png = {output_file_png} dpi={dpi}")
      plt.savefig(output_file_png,dpi=dpi)
   else:
      print(f"output_file_pdf = {output_file_pdf} dpi={dpi}")
      plt.savefig(output_file_pdf,dpi=dpi)
      plt.savefig(output_file_tiff)

plt.title(s_str, wrap=True)
##plt.clim(0, 1000)

plt.waitforbuttonpress()
plt.close()

if save_sp == 1:
   quit()

fig = plt.figure("Correlated Motion Vectors",figsize=(6,6))

plt.title(s_str, wrap=True)
plt.plot(Avg_Vect[:,1],Avg_Vect[:,0],'.k')

plt.axis([Cmin,Cmax,Rmax,Rmin])
plt.xlabel("$\\Delta$C",fontsize=18)
plt.ylabel("$\\Delta$R",fontsize=18)
plt.axvline(x = 0, linestyle = 'dashed', lw=1, color='k')
plt.axhline(y = 0, linestyle = 'dashed', lw=1, color='k')
##plt.text(0,0,"(0,0)")
plt.waitforbuttonpress()
plt.close()

fig, axs = plt.subplots(2,figsize=(16,9),sharex=True, sharey=True)
fig.suptitle("Separated Vectors of "+s_str,wrap=True)
axs[0].plot(Avg_Vect[:,0])
axs[0].set_title("Row Direction")
axs[1].plot(Avg_Vect[:,1])
axs[1].set_title("Column Direction")
if fixed_range_sp == 1:
   axs[0].set_ylim([-1*def_range,def_range])
else:
   if max(abs(p_max),abs(p_min)) < 10:
      axs[0].set_ylim([-10,10])
   
plt.waitforbuttonpress()
plt.close()
