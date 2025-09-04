import cv2
from matplotlib import pyplot as plt
import time
import numpy as np
import sys
import tifffile
from skimage import color, data, restoration
import math
from skimage import io
import os.path
import scipy.ndimage

Colormap=['gray','Greys','summer','winter','hot','bone','hsv']
selected_color = 0
disp_delay = 0.05

no_show_sp = 0

##########################################################   
argc_no = len(sys.argv)
if (argc_no < 2):
   print("No input!");
   print(" -i input file name");
   print(" -noblank turn off the blank checking procedure");
   print(" -x0, -y0, -x1, and -y1")
   print(" (x0,y0)------------------------------  ")
   print("    |                                |  ")
   print("    |                                |  ")
   print("    |                                |  ")
   print("    ------------------------------(x1,y1") 
   print(" -no_show directly output the cropping video")
   print(" -noshow directly output the cropping video")
   exit()
################# Set parameters ################
input_file = " "

start_frame = 0
end_frame = -1 

L_st = 0
L_end = 0
output_file = ""
o_sp = 0

x0 = 0
y0 =-1 
x1 = 0
y1 = -1
xy_sp = 0
######### Extract the input parameters from argv ###########
i=1
save_sp = 0
Nrange = 0

while (i<argc_no ):
   if (sys.argv[i] == "-i"):
      input_file = sys.argv[i+1]
      i += 2
   elif (sys.argv[i] == "-start"):
      L_st = (int)(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-end"):
      L_end = (int)(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-Nrange"):
      Nrange = (int)(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-o"):
      output_file = sys.argv[i+1]
      o_sp  = 1
      i += 2
   elif (sys.argv[i] == "-x0"):
      x0 = (int)(sys.argv[i+1]) 
      xy_sp = 1
      i += 2
   elif (sys.argv[i] == "-y0"):
      y0 = (int)(sys.argv[i+1]) 
      xy_sp = 1
      i += 2
   elif (sys.argv[i] == "-x1"):
      x1 = (int)(sys.argv[i+1])
      xy_sp = 1
      i += 2 
   elif (sys.argv[i] == "-y1"):
      y1 = (int)(sys.argv[i+1])
      xy_sp = 1
      i += 2
   elif (sys.argv[i] == "-save"):
      save_sp = 1
      i+=1
   elif (sys.argv[i] == "-no_show"):
      no_show_sp = 1
      i+=1
   elif (sys.argv[i] == "-noshow"):
      no_show_sp = 1
      i+=1
   else:
      i += 2
##################################################################################
###    Check input parameters ###########
##################################################################################
if (input_file == " "):
    print("No input")
    exit()

try:
   InFile = tifffile.TiffFile(input_file)
#except IOError:
except (FileNotFoundError, OSError) as e:
   print(f"\nOpen file {input_file} error! \nQuit")
   quit()
length = len(InFile.pages)

print("Single Frame Read-Process-Write (RPW) Mode")

#InFile = tifffile.TiffFile(input_file)
length = len(InFile.pages)
(Row, Col) = InFile.pages[0].shape
total_sum = Row*Col*length
ROW = Row; COL = Col
input_type = (str)(InFile.pages[0].dtype)

if xy_sp == 1:
   if (x1 < x0) or (x1 > ROW-1):
      x1 = ROW-1
   if (y1 < y0) or (y1 > COL-1):
      y1 = COL-1
   if (x0 < 0) or (x0 > x1):
      x0 = 0
   if (y0 < 0) or (y0 > y1):
      y0 = 0
else:
   x0 = (int)(ROW*0.25)
   x1 = (int)(ROW*0.75)
   y0 = (int)(COL*0.25)
   y1 = (int)(COL*0.75)

if L_end == 0:
   L_end = length

print("The total frame length",length)
print(f"data type = {input_type}")
print(f"Row = {Row}  Col = {Col}")
print("Start frame = ",L_st,"   end frame= ",L_end)
print("Inital cropping positions (",x0,",",y0,") - (",x1,",",y1,")")

if save_sp == 0:
   print("\nOperating Keys")
   print("\"d\": Play forward.  \"b\": Play backward. \"q\": Quit.")
   print("\"g\": Fast play forward.  \"n\": Fast play backward. ")
   print("\"p\": Increase Vmax. \"l\": Decrease Vmax. ")
   print("\"k\": Increase Vmin. \"m\": Decrease Vmin. ")
   print("\"S\": Save current image ")
   print("\"c\": Change color map.")
   print("")
   print("               ^          |               ")
   print("               |  \"up\"    V  \"i\"        ")
   print("   <- \"left\"                      \"right\" ->")
   print("   -> \"j\"                             \"k\" <-")
   print("               |  \"down\"  ^  \"m\"        ")
   print("               V          |               ")
   print("")

############ Detect Keyboard Event ###############
the_key = None
def press(event):
    global the_key
    the_key = event.key
##################################################

from matplotlib.backend_bases import MouseButton

i =0

if save_sp == 0:
   plt.figure(i+1,figsize=(8,6))
   plt.title(input_file)

   corp = "("+str(x0)+","+str(y0)+")-("+str(x1)+","+str(y1)+")"
   plt.suptitle(str(i+1)+" "+corp+'  ColorMap='+Colormap[selected_color])
   frame = InFile.pages[i].asarray()
   plt.imshow(frame,cmap=Colormap[selected_color])
   plt.colorbar()
   plt.connect('button_press_event', press)
   plt.close()

rec_color = np.max(InFile.pages[i].asarray())
step = 10

while (i<length+1) and save_sp==0:
   plt.close()
   
   plt.figure(input_file,figsize=(8,6))
   corp = "("+str(x0)+","+str(y0)+")-("+str(x1)+","+str(y1)+")"
   plt.suptitle(str(i+1)+" "+corp+'  ColorMap='+Colormap[selected_color])
   plt.title(str(i+1)+'/'+str(length))

   cp_image = np.copy(InFile.pages[i].asarray())
   cp_image[x0,y0:y1] = rec_color 
   cp_image[x1,y0:y1] = rec_color
   cp_image[x0:x1,y0] = rec_color
   cp_image[x0:x1,y1] = rec_color

   plt.imshow(cp_image,cmap=Colormap[selected_color])
   cbar = plt.colorbar()
   plt.pause(disp_delay)

   plt.gcf().canvas.mpl_connect('key_press_event', press)
   while not plt.waitforbuttonpress(): pass
   if (the_key =="d"):
      if (i<length-1):
         i = i + 1
   if (the_key =="g"):
      if (i<length-5):
         i = i + 5
   if (the_key =="b"):
      if (i>0):
         i = i - 1
   if (the_key =="c"):
         selected_color = (selected_color+1)%6
   if (the_key =="up"):
      x0 -= step
      if x0 < 0:
         x0 = 0
   if (the_key =="down"):
      x1 += step
      if x1 > ROW-1:
         x1 = ROW -1 
   if (the_key =="left"):
      y0 -= step
      if y0 <0:
         y0 = 0
   if (the_key =="right"):
      y1 += step
      if y1 > COL-1:
         y1 = COL - 1
   if (the_key =="i"):
      x0 += step
      if x0 > x1-1:
         x0 = x1-1
   if (the_key =="m"):
      x1 -= step
      if x1 < x0+1:
         x1 = x0+1
   if (the_key =="j"):
      y0 += step
      if y0 > y1-1:
         y0 = y1-1
   if (the_key =="k"):
      y1 -= step
      if y1 < y0-1:
         y1 = y0-1
   if (the_key =="S"):
      save_sp = 1
      plt.close()
      i = length + 1 
   if (the_key =="q"):
      plt.close()
      exit()
print("(x0,y0)-(x1,y1) = (",x0,",",y0,")-(",x1,",",y1,")")

if Nrange > 0:
   x0 = Nrange;        y0 = Nrange
   x1 = ROW - Nrange;  y1 = COL-Nrange
if save_sp == 1:
   if (o_sp == 0):
      if Nrange == 0:
         output_file = input_file[:-5] + "."
         output_file += str(x0)+"_"+str(y0)+"."
         output_file += str(x1)+"_"+str(y1)
         output_file += ".tiff"
      else:
         output_file = input_file[:-5] + "."
         output_file += "Nr"+str(Nrange)
         output_file += ".tiff"

   cp_row = x1 - x0
   cp_col = y1 - y0

   star_no = 100
   star_step = (int)((L_end-L_st)/star_no)
   for i in range(star_no):
      print(".",end="",flush=True)
   for i in range(star_no):
      print("\x08",end="",flush=True)

   with tifffile.TiffWriter(output_file,bigtiff=True) as tif:
      for i in range(L_st,L_end):
         if L_end-L_st > star_step:
            if (i % star_step == 0):
               print("@",end="",flush=True)   
         frame = InFile.pages[i].asarray()
         image = frame[x0:x1,y0:y1]
         tif.write(image,photometric='minisblack',contiguous=True)
   print("")
quit()

