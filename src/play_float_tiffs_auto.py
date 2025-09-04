import cv2
from matplotlib import pyplot as plt
import tifffile
import numpy as np
import sys,os
from skimage import io
import time

############## Keyboard event detection ############
pause_flag=0

##########################################################   
argc_no = len(sys.argv)
if (argc_no < 2):
   print("No input!");
   print(" -i input file name");
   print("[Options]")
   print(" -c selected color map (0:gray,1:Greys,2:summer,3:winter,4:hot,5:bone,6:hsv) ")
   print(" -blank check the blank frames")
   print(" -smooth (no parameter, turn on the median filter smoothing)")           
   exit()
################# Set parameters ################
input_file = " "
min_file = " "
min_file_sp = 0

#reference:https://matplotlib.org/stable/tutorials/colors/colormaps.html
Colormap=['gray','Greys','summer','winter','hot','bone','hsv']
selected_color = 0
color_no = 7
selected_frame = -1 
Vmax = -9999
Vmin = 9999
disp_delay = 0.05

Nrange = 50
c_radius = 50

blank_sp = 0 
smoothing_sp = 0
save_sp = 0
noshow_sp = 0
c_show = 0
fno_sp = 0
no_sp = -100

input_plus_min_file = " "
in_min_out_sp = 0

######### Extract the input parameters from argv ###########
i=1
while (i<argc_no ):
   if (sys.argv[i] == "-i"):
      input_file = sys.argv[i+1]
      i += 2
   elif (sys.argv[i] == "-min_file"):
      min_file = sys.argv[i+1]
      min_file_sp = 1
      i += 2
   elif (sys.argv[i] == "-in_min_out"):
      in_min_out_sp = 1
      i += 1
   elif (sys.argv[i] == "-input_plus_min_file"):
      input_plus_min_file = sys.argv[i+1]
      i += 2
   elif (sys.argv[i] == "-c_show"):
      c_show = 1
      i += 1
   elif (sys.argv[i] == "-Nrange"):
      Nrange = (int)(float(sys.argv[i+1]))
      i += 2
   elif (sys.argv[i] == "-c_radius"):
      c_radius = (int)(float(sys.argv[i+1]))
      i += 2
   elif (sys.argv[i] == "-no_sp"):
      no_sp = int(sys.argv[i+1])
      fno_sp = 1
      i += 2
   elif (sys.argv[i] == "-save"):
      save_sp = 1
      i += 1
   elif (sys.argv[i] == "-blank"):
      blank_sp = 1
      i += 1
   elif (sys.argv[i] == "-c"):
      selected_color= int(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-start"):
      selected_frame= int(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-vmax"):
      Vmax= int(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-vmin"):
      Vmin= int(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-delay"):
      disp_delay = float(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-smooth"):
      smoothing_sp = 1
      i += 1
   elif (sys.argv[i] == "-noshow"):
      noshow_sp = 1
      i += 1
   else:
      i += 2
######## Check input parameters ###########
if (input_file == " "):
    print("No input")
    exit()
if (selected_color <0 or selected_color>6):
    selected_color = 0

############ Detect Keyboard Event ###############
the_key = None
def press(event):
    global the_key
    the_key = event.key
##################################################

try:
   f_size = os.path.getsize(input_file)
except IOError:
   print(f"\nOpen file {input_file} error! \nQuit")
   quit()

f_size = (f_size/1000000000)
print(f"File size = {f_size:.2f} GB" )
      
InFile = tifffile.TiffFile(input_file)
length = len(InFile.pages)
(Row, Col) = InFile.pages[0].shape
ROW = Row; COL= Col
input_type = str(InFile.pages[0].dtype)

print(f"Shape of original video = {InFile.pages[0].shape}  Length = {length}")
print(f" Original data type = ", input_type)

if c_show == 1:
   min_image = InFile.pages[0].asarray()
   Image = min_image
   #from skimage import filters
   #from skimage.measure import regionprops
   #from skimage.measure import centroid
   #from skimage import measure
   from skimage import filters, io, measure
   from skimage.measure import regionprops
   threshold_value = filters.threshold_otsu(Image)
   labeled_foreground = (Image > threshold_value).astype(int)
   properties = regionprops(labeled_foreground, Image)
   Crow, Ccol = properties[0].centroid
   WCrow, WCcol = properties[0].weighted_centroid
   Crow = (int)(Crow); Ccol = (int)(Ccol)
   print(f"(Crow,Ccol) = ({Crow},{Ccol})")
########################################################

if (length == 1):
   print("single image file")

str_colormap = Colormap[selected_color]

print("Find the maximum and minimum values of all image frames")

if fno_sp > 0:
   if no_sp < 0 or fno_sp > length -1:
      fno_sp = 0
      no_sp = -100
      print(f"no_sp is out of range")
   else:
      frame = InFile.pages[no_sp].asarray()
      s_filename = input_file+"."+str(no_sp)
      if smoothing_sp == 1:
         frame = frame.astype(np.int16)
         frame = cv2.medianBlur(frame,5)
         s_filename += ".sm"
      bimage = frame.astype(input_type)
      tifffile.imwrite(s_filename+".tiff",bimage,photometric='minisblack')
      plt.imsave(s_filename+".png",bimage,cmap=Colormap[selected_color])
      quit()

fmax_sp = 0; fmin_sp = 0

if min(ROW,COL) < 2*Nrange:
   Nrange = (int)(0.1*(max(ROW,COL)- min(ROW,COL)))

print(f"Nrange = {Nrange}")

if Vmax >0  and Vmin < 9999:
   frame_max = Vmax
   frame_min = Vmin
else:
   star_no = 100
   star_step = (int)(length/star_no)

   for i in range(star_no):
      print(".",end="",flush=True)
   for i in range(star_no):
      print("\x08",end="",flush=True)

   frame_max = -99999.0
   frame_min = 99999.0
   for i in range(length):
      if (i % star_step == 0):
         print("@",end="",flush=True)

      frame = InFile.pages[i].asarray()
      if smoothing_sp == 1:
         if not input_type.startswith('float'):
            frame = cv2.medianBlur(frame,5)
      pframe = frame[Nrange:ROW-Nrange,Nrange:COL-Nrange]

      n_frame_max = max(frame_max,np.max(pframe))
      n_frame_min = min(frame_min,np.min(pframe))
      if n_frame_max > frame_max:
         frame_max = n_frame_max
         fmax_sp = i
      
      if n_frame_min < frame_min:
         frame_min = n_frame_min
         fmin_sp = i      
      ### print(f"n_frame_max = {n_frame_max} ")
   print("")

print(f"frame_max = {frame_max}")


IN_max = frame_max
IN_min = frame_min
pmax = IN_max
pmin = IN_min

"""
if IN_max < 65500:
   pmax = IN_max+2.0
else:
   pmax = IN_max

if IN_min >2:
   pmin = IN_min-2.0
else:
   pmin = IN_min
"""
print(f"Input file name: {input_file} max.={pmax} at {fmax_sp} and min.={pmin} at {fmin_sp}")
if min_file_sp >0:
   print("Input minimum file name:",min_file)
print("Total Frame number : ",length)
print("Image size => ",ROW,"x",COL)
print("Intensity Min.: ",IN_min,"   Max: ",IN_max)
print("Color map : ",Colormap[selected_color])
print("Display delay => ",disp_delay)

print("\nOperating Keys")
print("\"d\": Play forward.  \"b\": Play backward. \"q\": Quit.")
print("\"g\": Fast play forward.  \"n\": Fast play backward. ")
print("\"p\": Increase Vmax. \"l\": Decrease Vmax. ")
print("\"k\": Increase Vmin. \"m\": Decrease Vmin. ")
print("\"S\": Save current image ")
print("\"c\": Change color map.")

##################################################
if (selected_frame > -1):
   i = selected_frame
else:
   i = 0

from matplotlib.backend_bases import MouseButton

cv2.namedWindow(input_file)
color = (250,250,250)
speed = 1
s_rate = 0.05
sp = 1
##Colormap=['gray','Greys','summer','winter','hot','bone','hsv']

##############################################################################
### https://docs.opencv.org/4.x/d3/d50/group__imgproc__colormap.html
##############################################################################
color_map = [cv2.COLORMAP_BONE,cv2.COLORMAP_OCEAN,cv2.COLORMAP_HOT,cv2.COLORMAP_PARULA,cv2.COLORMAP_TWILIGHT,cv2.COLORMAP_PLASMA,cv2.COLORMAP_HSV]

c_map = ["BONE","OCEAN","HOT","PARULA","TWILIGHT","PLASMA","HSV"]

color_sp =0 
fintscale = 0.7
thickscale = 2

dpmax = pmax-pmin

i= 0
print("Functional keys:")
print(" c : change color (GRAYSCALE,BONE, OCEAN, HOT, PARULA, TWILIGHT, PLASMA)")
print(" i : increase speed ")
print(" d : decrease speed ")
print(" s : pause and push any key to continue")
print(" w : write the current frame to a single tiff file")
print(" q : quit ")

if min_file_sp == 1:
   min_image = io.imread(min_file)
   min_image = min_image.astype(np.int16)
   min_image = cv2.medianBlur(min_image,5)

stop_sp = 0
while stop_sp == 0:
   pframe = InFile.pages[i].asarray()
   pframe = pframe.astype(np.int16)

   if min_file_sp == 0:
      pframe = pframe - pmin
   else: 
      pframe = cv2.medianBlur(pframe,5)
      pframe = pframe - min_image
      pframe = cv2.medianBlur(pframe,5)
      pframe[pframe <0] = 0
   
   pframe = 255.*np.divide(pframe,dpmax)
   pframe[pframe<0] = 0; pframe[pframe>255] = 255
   frame = pframe.astype(np.uint8)
    
   frame = cv2.applyColorMap(frame, color_map[color_sp])

   show_txt = str(i) +" ["+c_map[color_sp]+"]"
   if speed >1 :
      show_txt += " x"+str(speed)
   if s_rate > 0.05 :
      show_txt += " slow"
   pframe = cv2.putText(frame, show_txt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, fintscale, color, thickscale, cv2.LINE_AA)
   
   if c_show == 1:
      radius = 320;  thickness = 2;  ## color = (255, 0, 0)
      frame = cv2.circle(frame, (Ccol,Crow), radius, color, thickness)
      frame = cv2.circle(frame, (Ccol,Crow), radius+80, color, thickness)

   cv2.imshow(input_file,frame)

   i = (i+speed) % length
   time.sleep(s_rate)

   waitKey = (cv2.waitKey(1) & 0xFF)
   if waitKey == ord('i'):
      speed *= 2
      s_rate = 0.05
   if waitKey == ord('d'):
      if speed > 1:
         speed = (int)(speed/2)
      else:
         s_rate *= 2
   if waitKey == ord('s'):
      cv2.waitKey(0)
   if waitKey == ord('w'):
      tifffile.imwrite(input_file+"."+str(i)+".tiff",InFile.pages[i].asarray(),photometric='minisblack')
   if waitKey == ord('S'):
      tifffile.imwrite(input_file+"."+str(i)+".tiff",InFile.pages[i].asarray(),photometric='minisblack')
   if waitKey == ord('c'):
      color_sp = (color_sp+1)%len(color_map)  
   if  waitKey == ord('q'): #if Q pressed you could do something else with other keypress
      print("closing video and exiting")
      cv2.destroyWindow(input_file)
      stop_sp = 1

print("Stop")
if in_min_out_sp == 1 and min_file_sp == 1:
   if input_plus_min_file == "":
      if input_file[-5:] == ".tiff":
         input_plus_min_file = input_file[:-5]
      if input_file[-4:] == ".tif":
         input_plus_min_file = input_file[:-4]
      input_plus_min_file
   input_plus_min_file += ".p_min.tiff"

   InMinFile = tifffile.TiffFile(min_file)
   (Row, Col) = InMinFile.pages[0].shape
   min_type = (str)(InMinFile.pages[0].dtype)
   min_frame = InFile.pages[0].asarray()
   min_frame = min_frame.astype('int')

   star_step = 100
   neg_sp = 0
   neg_min = 99999 
   level_up = 1

   if level_up == 1:
      for i in range(length):
         if (i % star_step == 0):
            print("c",end="",flush=True)
         frame = InFile.pages[i].asarray()
         frame = frame.astype('int')

         pframe = frame - min_frame
         tmp_min = np.min(pframe)
         if tmp_min < neg_min:
            neg_min = tmp_min

   print("\n")
   with tifffile.TiffWriter(input_plus_min_file,bigtiff=True) as tif:
      for i in range(length):
         if (i % star_step == 0):
            print("w",end="",flush=True)
         frame = InFile.pages[i].asarray()
         pframe = frame - min_frame -neg_min
         pframe = pframe.astype(input_type)
         tif.write(pframe,photometric='minisblack',contiguous=True)
   print("")

