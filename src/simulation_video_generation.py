import cv2
import tifffile
import time
from skimage import io
import os.path

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle 
import numpy as np
import sys
import random

from skimage import filters
from skimage.measure import regionprops
from skimage.measure import centroid
from skimage import measure

from skimage.filters import butterworth

the_key = None
def press(event):
   global the_key
   the_key = event.key

Colormap=['gray','Greys','ocean_r','winter','hot','bone','viridis','magma','hsv']
color_sp = 5 ## cmap=Colormap[color_sp]

min_bk_file = " "
min_fov_file = " "

bk_file = " "
neuron_file = " "
output_file = " "
suffix_file_name = " "
mv_file = " "

fov_cen_row= 393;  fov_cen_col = 572
fov_radius = 345;  fov_drange  = 50

Oh_cen_row = 396;   Oh_cen_col = 572
Oh_radius  = 300;   Oh_drange  = 50

fov_box_r0 = 48;    fov_box_c0 = 227;
fov_box_r1 = 738;   fov_box_c1 = 917

ni_cen_row = 0;       ni_cen_col = 0
Up_no = 0
bk_select_region = 0
ns_select_region = 0

bk_video_max = -999
neuron_video_max = -999
signal_adjusted_ratio = 1.0

shaking_type = 1 ## 0:no shaking; 1: neuron shaking; 2: bk shaking; 3: both shaking 
shaking_sp = 0
random_gen_sp = 0
pv_mean = 0.0
pv_sigma = 1.0

cf_r = 0.01

show_sp = 0
show_mv_dist_sp = 0

out_length = 0
start = 10

argc_no = len(sys.argv)
if (argc_no < 2):
   print(f"")
   print(f"This program generates a simulated video: bk_video (fixed) + (neuron_f (shaking)) * OHole_map")
   print(f"For example:\n")
   print(f"python ./src/simulation_video_generation.py -bk_video CC1_2345_128.smBK.tiff -neuron_video train.04.00.smBP.tiff -bk_video_max 2793 -neuron_video_max 2710 -Up_no 2 -up_ni_cen_row 1124 -up_ni_cen_col 1474 -mv_file CC1.BP.tiff.v5.Lmv.vsum.8.3.5 -signal_adjusted_ratio 0.2 -length 1500 -prefix_name xxx -suffix_name v0")
   print(f"\nOutputs: xxx.v0.tiff, xxx.v0.tiff.info")
   print(f"[Major inputs]")
   print(f"  -bk_file : bk_video    (Background video: *.smBK.tiff)")
   print(f"  -neuron_file : neuron_f (spking neuron signal video)")
   print(f"  -length => output length of simulation video")
   print(f" ")
   print(f"[Parameters for bk_file]")
   print(f"  -ov_box_r0, -fov_box_c0, -fov_box_r1,-fov_box_c1 : region selection in bk video")
   print(f" ")
   print(f"[Parameters for OHole_map]")
   print(f"  -Oh_cen_row, -Oh_cen_col : center of OHole_map => clear obseratory region")
   print(f"  -Oh_radius and -Oh_drange : radius and degraded range of OHole_map")
   print(f" ")
   print(f"[Parameters for FOV in background video]")
   print(f"  -fov_cen_row and -fov_cen_col : center of FOV")
   print(f"  -fov_radius and -fov_drange : radius and degraded range of FOV")
   print(f" ")
   print(f"[Parameters for neuron_file: ]")
   print(f"  -up_ni_cen_row and -up_ni_cen_col : re-centerized of neuron_file")
   print(f"  -Up_no : number for up-sampling")
   print(f" ")
   print(f"[Parameters for signal adjustation] ")
   print(f"  -bk_video_max : the max intensity value of background video") 
   print(f"  -neuron_video_max : the max intensity value of neuron video")
   print(f"  -signal_adjusted_ratio : ratio for adjusting neuron signals in simulation video")
   print(f" ")
   print(f"[Parameters for generating random motion vectors:]")
   print(f"  -mv_file (optional) => motion vector file ")
   print(f"  -pv_mean and -pv_sigma : mean and sigma for generating random distribution")
   print(f" ")
   print(f"[Manual set selected region:]")
   print(f"  -bk_region_selection : select the background region") 
   print(f"  -ns_region_selection : select the neuron signal region") 
   print(f" ")
   print(f"[General options]")
   print(f"  -show    : show detailed results")
   print(f"  -show_mv : show the distribution of motion vectors")
   print(f"  -color   : selected color")
   print(f"  -prefix_name : prefix output video name ")
   print(f"  -suffix_name : suffix name add to prefix name")
   print(f"  -shaking_type: 0:____ None ; 1: ns__ neuron (default),2: __bk, 3: nsbk both shaking")
   print(f"  -fixed_bk no : use the no'th frame as bk")
   exit()

info_sp = 1
fixed_bk_sp = 0
fixed_bk_no = 100

prefix_name ="zzz"
suffix_name ="v0"
i=1
while (i<argc_no ):
   if (sys.argv[i] == "-bk_video"):
      bk_file = sys.argv[i+1]
      i += 2
   elif (sys.argv[i] == "-neuron_video"):
      neuron_file = sys.argv[i+1]
      i += 2
   elif (sys.argv[i] == "-bk_video_max"):
      bk_video_max = (int)(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-neuron_video_max"):
      neuron_video_max = (int)(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-signal_adjusted_ratio"):
      signal_adjusted_ratio = (float)(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-fov_cen_row"):
      fov_cen_row = (int)(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-fov_cen_col"):
      fov_cen_col = (int)(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-Oh_cen_row"):
      Oh_cen_row = (int)(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-Oh_cen_col"):
      Oh_cen_col = (int)(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-up_ni_cen_row"):  ## row center at the upsampling video
      ni_cen_row = (int)(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-up_ni_cen_col"):  ## column center at the upsampling video
      ni_cen_col = (int)(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-fov_radius"):
      fov_radius = (int)(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-fov_drange"):
      fov_drange = (int)(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-Oh_radius"):
      Oh_radius = (int)(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-Oh_drange"):
      Oh_drange = (int)(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-fov_box_r0"):
      fov_box_r0 = (int)(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-fov_box_c0"):
      fov_box_c0 = (int)(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-fov_box_r1"):
      fov_box_r1 = (int)(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-fov_box_c1"):
      fov_box_c1 = (int)(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-Up_no"):
      Up_no = (int)(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-length"):
      out_length = (int)(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-mv_file"):
      shaking_sp = 1
      mv_file = sys.argv[i+1]
      i += 2
   elif (sys.argv[i] == "-shaking_type"): ## 0: No shaking; 1: neuron shaking
      shaking_type = (int)(sys.argv[i+1]) ## 2: bk shaking; 3: both simul. skaking
      if shaking_type == 0:
         shaking_sp = 0
      i += 2
   elif (sys.argv[i] == "-pv_sigma"):
      shaking_sp = 1
      pv_sigma = (float)(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-pv_mean"):
      shaking_sp = 1
      pv_mean = (float)(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-ns_region_selection"):
      ns_select_region = 1
      i += 1
   elif (sys.argv[i] == "-bk_region_selection"):
      bk_select_region = 1
      i += 1
   elif (sys.argv[i] == "-show"):
      show_sp = 1
      i += 1
   elif (sys.argv[i] == "-show_mv"):
      show_mv_dist_sp = 1
      i += 1
   elif (sys.argv[i] == "-color"):
      color_sp = (int)(sys.argv[i+1])%len(Colormap)
      i += 2
   elif (sys.argv[i] == "-prefix_name"):
      prefix_name = sys.argv[i+1]
      i += 2
   elif (sys.argv[i] == "-suffix_name"):
      suffix_name = sys.argv[i+1]
      i += 2
   elif (sys.argv[i] == "-fixed_bk"):
      fixed_bk_no = (int)(sys.argv[i+1])
      fixed_bk_sp = 1
      i += 2
   elif (sys.argv[i] == "-start"):
      start = (int)(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-info"):
      info_str = sys.argv[i+1]
      if info_str == "off":
         info_sp = 0
      if info_str == "on":
         info_sp = 1
      i += 2
   else:
      i += 1

##############################################################################################
###  Reading background video
##############################################################################################
try:
   bk_video = tifffile.TiffFile(bk_file)
except IOError:
   print(f"\nCannot open video file \"{bk_file}\" error! \nQuit")
   quit()

bk_video_len = len(bk_video.pages)
(bk_video_Row, bk_video_Col) = bk_video.pages[0].shape
bk_video_type = str(bk_video.pages[12].dtype)
bkV_image = bk_video.pages[1].asarray()

print(f"BK Video {bk_file} => ({bk_video_Row},{bk_video_Col})  L = {bk_video_len} data type = {bk_video_type}")

if out_length > bk_video_len:
   out_length = bk_video_len

#############################################################################################
### Select region in input bk video
#############################################################################################
if bk_select_region == 1:
   Image = bk_video.pages[1].asarray()
   (min_fov_f_Row, min_fov_f_Col) = Image.shape

   fig = plt.figure(bk_file,figsize=(8,6))
   """
   Oh_cen_row = (int)(min_fov_f_Row/2) ;  Oh_cen_col = (int)(min_fov_f_Col/2)
   fov_cen_row = (int)(min_fov_f_Row/2); fov_cen_col = (int)(min_fov_f_Col/2)

   fov_box_r0 = 0;   fov_box_c0 = 0
   fov_box_r1 = min_fov_f_Row-1;   fov_box_c1 = min_fov_f_Col-1
   """
   width  = fov_box_c1 - fov_box_c0
   height = fov_box_r1 - fov_box_r0

   sel_sp = 1

   sel_fov_sp = 1

   while (sel_sp == 1):
      plt.clf()
      box_str="  ("+str(fov_box_r0)+","+str(fov_box_c0)+") - ("+str(fov_box_r1)+","+str(fov_box_c1)+")"
      plt.suptitle("  ColorMap="+Colormap[color_sp] + "  Box: "+box_str)
      if sel_fov_sp == 0:
         p_cen = "("+str(Oh_cen_row)+","+str(Oh_cen_col)+")"
         plt.title("Oh_rad="+str(Oh_radius)+" Oh center="+p_cen)
      else:
         p_cen = "("+str(fov_cen_row)+","+str(fov_cen_col)+")"
         plt.title("fov_rad="+str(fov_radius)+" fov center="+p_cen)
      cp_image = np.copy(Image)
  
      plt.imshow(cp_image,cmap=Colormap[color_sp]) 
      circle1 = plt.Circle((Oh_cen_col,Oh_cen_row),Oh_radius,fill=False,color='r',lw=2)
      circle2 = plt.Circle((fov_cen_col,fov_cen_row),fov_radius,fill=False,color='y',lw=2)

      width  = fov_box_c1 - fov_box_c0; height = fov_box_r1 - fov_box_r0
      plt.gca().add_patch(circle1)
      plt.gca().add_patch(circle2)

      plt.gca().add_patch(Rectangle((fov_box_c0,fov_box_r0),width,height,
                                 fc = 'none',
                                 ec = 'g',
                                 lw = 2))
      cbar = plt.colorbar()
      plt.pause(0.01)

      plt.gcf().canvas.mpl_connect('key_press_event', press)
      while not plt.waitforbuttonpress(): pass
      if (the_key =="c"):
         color_sp = (color_sp+1)%len(Colormap)
      if (the_key == "up"):
         if sel_fov_sp == 0:
            if Oh_cen_row - 1 > 0: 
               Oh_cen_row -= 1
         else:
            if fov_cen_row - 1 > 0:
               fov_cen_row -= 1
      if (the_key == "down"):
         if sel_fov_sp == 0:
            if Oh_cen_row + 1 < min_fov_f_Row:
               Oh_cen_row += 1
         else:
            if fov_cen_row + 1 < min_fov_f_Row:
               fov_cen_row += 1
      if (the_key == "left"):
         if sel_fov_sp == 0:
            if Oh_cen_col - 1 > 0:
               Oh_cen_col -= 1
         else:
            if fov_cen_col - 1 > 0:
               fov_cen_col -= 1
      if (the_key == "right"):
         if sel_fov_sp == 0:
            if Oh_cen_col + 1 < min_fov_f_Col:
               Oh_cen_col += 1
         else:
            if fov_cen_col + 1 < min_fov_f_Col:
               fov_cen_col += 1
      if (the_key == "i"):
         if sel_fov_sp == 0:
            Oh_radius += 1
         else:
            fov_radius += 1
      if (the_key == "d"):
         if sel_fov_sp == 0:
            if Oh_radius - 1 > 0:
               Oh_radius -= 1
         else:
            if fov_radius - 1 > 0:
               fov_radius -= 1
      if (the_key == "w"):
         sel_fov_sp = (sel_fov_sp+1)%2
      if (the_key =="q"):
         sel_sp = 0
         plt.close()

      if fov_cen_row - fov_radius < 0:
         fov_box_r0 = 0
      else:
         fov_box_r0 = fov_cen_row - fov_radius

      if fov_cen_row + fov_radius > min_fov_f_Row:
         fov_box_r1 = min_fov_f_Row
      else:
         fov_box_r1 = fov_cen_row + fov_radius

      if fov_cen_col - fov_radius < 0:
         fov_box_c0 = 0
      else:
         fov_box_c0 = fov_cen_col - fov_radius

      if fov_cen_col + fov_radius > min_fov_f_Col:
         fov_box_c1 = min_fov_f_Col
      else:
         fov_box_c1 = fov_cen_col + fov_radius

fov_row = fov_box_r1 - fov_box_r0
fov_col = fov_box_c1 - fov_box_c0

fov_cen_row = fov_cen_row - fov_box_r0
fov_cen_col = fov_cen_col - fov_box_c0

(bk_fov_r_st,bk_fov_r_ed) = (fov_box_r0,fov_box_r1)
(bk_fov_c_st,bk_fov_c_ed) = (fov_box_c0,fov_box_c1)

Oh_cen_row = Oh_cen_row - fov_box_r0
Oh_cen_col = Oh_cen_col - fov_box_c0

out_row = bk_fov_r_ed-bk_fov_r_st
out_col = bk_fov_c_ed-bk_fov_c_st

print(f" Ohole  ({Oh_cen_row},{Oh_cen_col}) Oh_radius = {Oh_radius}")
print(f" FOV    ({fov_cen_row},{fov_cen_col}) fov_radius = {fov_radius}")
print(f" BOX    ({fov_box_r0},{fov_box_c0}) - ({fov_box_r1},{fov_box_c1})")

#############################################################################################

### Ratio maps
FOV_map = np.zeros((fov_row,fov_col),dtype=float)
nonFOV_map = np.zeros((fov_row,fov_col),dtype=float)
OHole_map = np.zeros((fov_row,fov_col),dtype=float)
nonOHole_map = np.zeros((fov_row,fov_col),dtype=float)

fov_in_radius = fov_radius - fov_drange
Oh_in_radius  = Oh_radius - Oh_drange

for i in range(fov_row):
   ii = (i-fov_cen_row)*(i-fov_cen_row)
   for j in range(fov_col):
      jj = (j-fov_cen_col)*(j-fov_cen_col)
      rr = np.sqrt(ii+jj)
      if rr < fov_radius:
         if rr < fov_in_radius:
            FOV_map[i,j] = 1.0
         else:
            FOV_map[i,j] = 1.0 - (rr - fov_in_radius)/fov_drange
      else:
         FOV_map[i,j] = 0.0

      nonFOV_map[i,j] = 1.0 - FOV_map[i,j]

for i in range(fov_row):
   ii = (i-Oh_cen_row)*(i-Oh_cen_row)
   for j in range(fov_col):
      jj = (j-Oh_cen_col)*(j-Oh_cen_col)
      rr = np.sqrt(ii+jj)      
      if rr < Oh_radius:
         if rr < Oh_in_radius:
            OHole_map[i,j] = 1.0
         else:
            OHole_map[i,j] = 1.0 - (rr - Oh_in_radius)/Oh_drange
      else:
         OHole_map[i,j] = 0.0
      nonOHole_map[i,j] = 1.0 - OHole_map[i,j] 

if show_sp == 1:
   plt.figure(min_bk_file,figsize=(8,6))
   plt.title("FOV_map")
   plt.imshow(FOV_map,cmap=Colormap[color_sp])
   circle1 = plt.Circle((fov_cen_col,fov_cen_row),fov_radius,color='y',lw=2,fill=False)
   plt.gca().add_patch(circle1)
   plt.waitforbuttonpress()
   plt.close()

   plt.figure(min_bk_file,figsize=(8,6))
   plt.title("OHole_map")
   plt.imshow(OHole_map,cmap=Colormap[color_sp])
   circle1 = plt.Circle((Oh_cen_col,Oh_cen_row),Oh_radius,fill=False,color='y',lw=2)
   plt.gca().add_patch(circle1)
   plt.waitforbuttonpress()
   plt.close()

##################################################################################
### Reading the neuron spiking file
##################################################################################
neuron_file
try:
   neuron_f = tifffile.TiffFile(neuron_file)
except IOError:
   print(f"\nCannot open neuron file \"{neuron_file}\" error! \nQuit")
   quit()

neuron_len = len(neuron_f.pages)
(neuron_f_Row, neuron_f_Col) = neuron_f.pages[0].shape
neuron_f_type = str(neuron_f.pages[0].dtype)
neuron_image = neuron_f.pages[50].asarray()
print(f"{neuron_file} => ({neuron_f_Row},{neuron_f_Col})  L = {neuron_len} data type = {neuron_f_type}")

if (out_length == 0) or (out_length > neuron_len):
   out_length = neuron_len

###################################################################################
######## Check whether the neuron image needs to be upsampling #########
###################################################################################

niC_radius = fov_radius

if Up_no == 0:
   if min(neuron_f_Row,neuron_f_Col) < 2*fov_radius:
      Up_no = (int)((min(neuron_f_Row,neuron_f_Col)/(2*fov_radius)) +1)
      #### Upsampling
      if Up_no > 0:
         for i in range(Up_no):
            neuron_image = cv2.pyrUp(neuron_image)
      ##############$$$#####################################################################
      ### pyrUp()...performs the upsampling step of the Gaussian pyramid construction...
      ### First, it upsamples the source image by injecting even zero rows and columns and
      ### then convolves the result with the same kernel as in pyrDown() multiplied by 4.
      ######################################################################################
else:
   for i in range(Up_no):
      neuron_image = cv2.pyrUp(neuron_image)

###########################################################################################
#### Select the neuron region of interested
#### Row, Col, and Center of upsamapled neuron video
#### (niC_row,niC_col): center of neuron video and FOV region
#### [st_row:ed_row,st_col:ed_col]: FOV region
############################################################################################

(ni_row, ni_col) = neuron_image.shape

if ni_cen_row == 0 and ni_cen_col == 0:
   niC_row = (int)(ni_row/2);   niC_col = (int)(ni_col/2)
else:
   niC_row = ni_cen_row; niC_col = ni_cen_col

if Up_no > 0:
   d_size = pow(2,Up_no)

x0 = niC_row - fov_radius; y0 = niC_col - fov_radius
x1 = niC_row + fov_radius; y1 = niC_col + fov_radius

if ns_select_region == 0:
   print(f"Deafult center of neuron signal image ({niC_row},{niC_col})")
else:
   selected_color = color_sp
   step = 5*Up_no
   i = 100
   rec_color = np.max(neuron_f.pages[i].asarray())
   fig = plt.figure(neuron_file,figsize=(8,6))
   while (i<neuron_len+1):
      plt.clf()
      corp = "("+str(x0)+","+str(y0)+")-("+str(x1)+","+str(y1)+")"
      p_cen = "("+str(niC_row)+","+str(niC_col)+")"
      plt.suptitle(str(i+1)+" "+'  ColorMap='+Colormap[selected_color] +"  Center "+p_cen)
      tit_str = str(i+1)+'/'+str(neuron_len)+" Center = "+p_cen 
      tit_str += " radius = "+str(fov_radius)
      plt.title(tit_str)

      cp_image = np.copy(neuron_f.pages[i].asarray())
      for j in range(Up_no):
         cp_image = cv2.pyrUp(cp_image)

      fov_row_2 = (int)(fov_row/2); fov_col_2 = (int)(fov_col/2)

      x0 = niC_row - fov_row_2; y0 = niC_col - fov_col_2
      x1 = niC_row + fov_row_2; y1 = niC_col + fov_col_2
      plt.imshow(cp_image,cmap=Colormap[selected_color])
      circle1 = plt.Circle((niC_col,niC_row),Oh_radius,fill=False,color='y',lw=2)
      plt.gca().add_patch(circle1)
      width = x1-x0;  height=y1-y0
      plt.gca().add_patch(Rectangle((y0,x0),height,width,
                                 fc = 'none',
                                 ec = 'g',
                                 lw = 2))
      cbar = plt.colorbar()
      plt.pause(0.01)

      plt.gcf().canvas.mpl_connect('key_press_event', press)
      while not plt.waitforbuttonpress(): pass

      if (the_key =="n"):
         if (i<neuron_len-1):
           i = i + 1
      if (the_key =="N"):
         if (i<neuron_len-5):
            i = i + 5
      if (the_key =="b"):
         if (i>0):
            i = i - 1
      if (the_key =="B"):
         if (i>neuron_len+5):
            i = i - 5
      if (the_key =="c"):
            selected_color = (selected_color+1)%6
      if (the_key =="up"):
         if niC_row - fov_radius > -1*step:
            niC_row -= step
      if (the_key =="down"):
         if niC_row - fov_radius < ni_row-2-step:
            niC_row += step
      if (the_key =="left"):
         if niC_col - fov_radius > -1*step:
            niC_col -= step
      if (the_key =="right"):
         if niC_col + fov_radius < ni_col-2-step:
            niC_col += step
      if (the_key == "w"):
         n_img = cp_image[x0:x1,y0:y1]
         n_img_file = "z_img."+str(i)+".tiff"
         tifffile.imwrite(n_img_file,n_img,photometric='minisblack')
         print(f"Write the selected region to {n_img_file}")
      if (the_key =="q"):
         i = neuron_len+ 1
         plt.close()

   print(f"Selected center => ({niC_row},{niC_col})")
#############################################################################################
### Selected region as FOV in the upsampling neuron video 2xFOV_radius X 2xFOV_radius
#############################################################################################
fov_row_2 = (int)(fov_row/2);        fov_col_2 = (int)(fov_col/2)

(ni_st_row,ni_st_col) = (niC_row - fov_row_2,niC_col - fov_col_2)
(ni_ed_row,ni_ed_col) = (niC_row + fov_row_2,niC_col + fov_col_2)

###########################################################################################
## Calculate the ratio for adjusting the neruon signal levels as those in real video
###########################################################################################

star_no = 100
star_step = (int)(bk_video_len/star_no)

if bk_video_max > 0:
   print(f"Precalculated maximun value of {bk_file} = {bk_video_max}")
   bk_max = bk_video_max
else:
   print(f"Calculate the maximun value of {bk_file}\n")
   for i in range(star_no):
      print(".",end="",flush=True)
   for i in range(star_no):
      print("\x08",end="",flush=True)

   bk_max = -9999
   for i in range(bk_video_len):
      if (i % star_step == 0):
         print("@",end="",flush=True)
      bk_tmp = np.max(bk_video.pages[i].asarray())
      if bk_tmp > bk_max:
         bk_max = bk_tmp
   print(f"\nMax of {bk_file} = {bk_max}")

if neuron_video_max > 0:
   print(f"Precalculated maximun value of {neuron_file} = {neuron_video_max}")
   Neu_Sig_max = neuron_video_max
else:
   print(f"Calculate the maximun value of {neuron_file}\n")
   star_step = (int)(neuron_len/star_no)
   for i in range(star_no):
      print(".",end="",flush=True)
   for i in range(star_no):
      print("\x08",end="",flush=True)

   Neu_Sig_max = -99999
   for i in range(neuron_len):
      if (i % star_step == 0):
         print("@",end="",flush=True)
      bk_tmp = np.max(neuron_f.pages[i].asarray())
      if bk_tmp > Neu_Sig_max:
         Neu_Sig_max = bk_tmp

   print(f"\nMax of {neuron_file} = {Neu_Sig_max}")

n_signal_ratio = bk_max/Neu_Sig_max
print(f"n_signal_ratio = {n_signal_ratio}")

n_signal_ratio = signal_adjusted_ratio*n_signal_ratio
print(f"Adjusted Normalized ratio n_signal_ratio = {n_signal_ratio}")

if show_sp == 1:
   ##################################################################
   fov_bk_img = bk_video.pages[1].asarray() 
   fov_bk_img = fov_bk_img[bk_fov_r_st:bk_fov_r_ed,bk_fov_c_st:bk_fov_c_ed]

   fov_neuron_img = neuron_image[ni_st_row:ni_ed_row,ni_st_col:ni_ed_col]
   fov_neuron_img = butterworth(fov_neuron_img, cf_r, True, 8)

   aaa = n_signal_ratio*np.multiply(fov_neuron_img,OHole_map)
   aaa = np.multiply(aaa,OHole_map)
   fov_bk_neuron_img = fov_bk_img + aaa.astype(int)

   plt.figure(neuron_file,figsize=(8,6))
   plt.title(neuron_file+"  fov_bk_img" )
   plt.imshow(fov_bk_img,cmap=Colormap[color_sp])
   plt.waitforbuttonpress()
   plt.close()

   plt.figure(neuron_file,figsize=(8,6))
   plt.title(neuron_file+"  fov_neuron_img" )
   plt.imshow(fov_neuron_img,cmap=Colormap[color_sp])
   plt.waitforbuttonpress()
   plt.close()

   plt.figure(neuron_file,figsize=(8,6))
   plt.title(neuron_file+"fov_bk_neuron_img" )
   plt.imshow(fov_bk_neuron_img,cmap=Colormap[color_sp])
   plt.waitforbuttonpress()
   plt.close()

##############################################################################################
## Read input motion vectors or generate random motion vectors 
##############################################################################################

if shaking_sp == 1:
   random_gen_sp = 1
   try:
      f = open(mv_file,'r')
      print(f"Generate shaking motion vectors by file {mv_file}")
      mv_count = 0
      random_gen_sp = 0
      for line in f:
         if (line[0] != '#') and (line[0] != '*'):
            fary = line.rstrip().split(',')
            if len(fary) == 10:
               mv_count += 1
      if mv_count < neuron_len:
         random_gen_sp = 1
      else:
         pv_row = np.zeros(mv_count,dtype=int)
         pv_col = np.zeros(mv_count,dtype=int)

         Avg_Vect = np.zeros((mv_count,2),dtype=int)
         trgP = np.zeros((mv_count,4),dtype=int)
         imgP = np.zeros((mv_count,4),dtype=int)
         q = 0
         f.seek(0)
         for line in f:
            if (line[0] != '#') and (line[0] != '*'):
               fary = line.rstrip().split(',')
               if len(fary) == 10:
                  Avg_Vect[q,0] = (int)(fary[0]); Avg_Vect[q,1] = (int)(fary[1])
                  pv_row[q] = (int)(fary[0]); pv_col[q] = (int)(fary[1]) 
                  ##pv_row[q] = -1*pv_row[q]  ; pv_col[q] = -1*pv_col[q]
                  for qq in range(4):
                     trgP[q,qq]=fary[qq+2]
                     imgP[q,qq]=fary[qq+6]
                  q += 1
      if show_sp > 0:
         plt.figure(mv_file,figsize=(6,6))
         plt.title(mv_file)
         plt.plot(Avg_Vect[:,1],Avg_Vect[:,0],'.k')
         plt.axis([-40,40,-40,40])
         plt.waitforbuttonpress()
         plt.close()
   except IOError:
      print(f"Simulation shaking scale by random function")
   if random_gen_sp == 1:
      pv_row = np.random.normal(pv_mean,pv_sigma,neuron_len).astype(int)
      pv_col = np.random.normal(pv_mean,pv_sigma,neuron_len).astype(int)

   color_mp = 'viridis'
   if show_mv_dist_sp == 1:
      mv_list = []
      for i in range(len(pv_row)):
         mv_list.append(((int)(pv_row[i]),(int)(pv_col[i])))
      mv_set = list(set(mv_list))
      mv_set_count = []
      point_count = np.zeros((len(mv_set),3),dtype=int)
      for i in range(len(mv_set)):
         (point_count[i,0],point_count[i,1]) = mv_set[i] 
         point_count[i,2] = mv_list.count(mv_set[i])      

      p_max = max(np.max(abs(pv_row)),np.max(abs(pv_col)))
      p_max = (int)(((int)(p_max/10.0)+1)*10)
      if p_max < 20:
         p_max = 20
      ##fig = plt.figure("Motion Vectors",dpi=dpi)
      fig = plt.figure("Motion Vectors",figsize=(8,6))
      plt.axis([-1*p_max,p_max,-1*p_max,p_max])
      plt.scatter(point_count[:,1],point_count[:,0],c=point_count[:,2],cmap=color_mp)
      plt.colorbar()
      plt.axvline(x = 0, linestyle = 'dashed', lw=1, color='k')
      plt.axhline(y = 0, linestyle = 'dashed', lw=1, color='k')
      plt.xlabel("$\\Delta$C",fontsize=18)
      plt.ylabel("$\\Delta$R",fontsize=18)
      plt.tight_layout()
      plt.waitforbuttonpress()
      plt.close()

output_file = prefix_name

if random_gen_sp == 1:
   pv_int = (int)(pv_sigma)
   output_file += ".pv"+str(pv_int)
else:
   if shaking_type == 0:
      output_file += ".pv0"
   else:
      output_file += ".mv"

if fixed_bk_sp == 1:
   output_file += ".bk"+str(fixed_bk_no)
else:
   output_file += ".bk"+str(start)+"_"+str(out_length+start)

if shaking_type == 0:
   output_file += ".____"  ## No shake neuron image and fix background image
if shaking_type == 1:
   output_file += ".ns__"  ## Shake neuron image and fix background image
if shaking_type == 2:
   output_file += ".__bk"  ##   Fix neuron image and shake background image
if shaking_type == 3:
   output_file += ".nsbk"  ## Shake both neuron and background images

output_file += "."+suffix_name+".tiff"

if info_sp == 1:
   output_info_file = output_file +".info"
   print(f"Write the execution information to {output_info_file}")
   import datetime
   with open(output_info_file, "w") as f:
      x = str(datetime.datetime.now())+"\n"
      x += 'python '+' '.join(sys.argv) 
      x += '\n\n'
      f.write(x)
      with open(sys.argv[0],'r') as s_code:
         lines = s_code.readlines()
      for line in lines:
         f.write(line)
   
##############################################################################
## bk_image     : an original bk image(frame) in input background video
## bk_fov_image : selected region in bk image
## neuron_image : an original neuron image
##              : a upsampling and smoothing neuron image
##              : high-pass filtering image
##              : shaking image
## neuron_fov_img : selected region in processed neuron image
##                : intensity adjustation
## neuron_fov_hole_img : filter out the intensity values out of the hole region
## bk_fov_neuron_hole_img : selected neuron region mixed with bk region
##############################################################################

m_range = 30

print(f"Output file {output_file}")
with tifffile.TiffWriter(output_file,bigtiff=True) as tif:
   for i in range(start,out_length+start):
      if (i % star_step == 0):  print("x",end="",flush=True)

      #########################################################################################
      ## Read the background video
      #########################################################################################
      if fixed_bk_sp == 0:
         bk_image = bk_video.pages[i].asarray().astype(int)
      else:
         bk_image = bk_video.pages[fixed_bk_no].asarray().astype(int)

      #########################################################################################
      ### Read neuron frame, upsampling, and translate
      #########################################################################################
      
      neuron_image = neuron_f.pages[i].asarray()
      for k in range(Up_no):
         neuron_image = cv2.pyrUp(neuron_image)    
         neuron_image = cv2.GaussianBlur(neuron_image, (11,11), 0)
      ## neuron_image = butterworth(neuron_image, cf_r, True, 8) ##
      neuron_image = neuron_image.astype(int)
     
      if shaking_sp == 1:
         dr = pv_row[i];  dc = pv_col[i]
         
         if shaking_type == 1 or shaking_type == 3:         
            neuron_image = np.roll(neuron_image,(dr,dc), axis = (0,1))
         
         if shaking_type == 2 or shaking_type == 3:
            bk_image = np.roll(bk_image,(dr,dc), axis = (0,1))
         
      bk_fov_image = bk_image[bk_fov_r_st:bk_fov_r_ed,bk_fov_c_st:bk_fov_c_ed]
      
      neuron_fov_img = neuron_image[ni_st_row:ni_ed_row,ni_st_col:ni_ed_col]

      neuron_fov_img = n_signal_ratio*neuron_fov_img

      ##neuron_fov_img = py.multiply(neuron_fov_img,FOV_map)
      neuron_fov_hole_img = np.multiply(neuron_fov_img,OHole_map)
      
      ##bk_fov_image = (py.multiply(bk_fov_image,FOV_map)).astype(int)

      ##### Mix neuron and bk BW images ####
     
      bk_fov_neuron_hole_img = bk_fov_image.astype(int)+ neuron_fov_hole_img.astype(int)
      bk_fov_neuron_hole_img = neuron_fov_hole_img.astype('uint16')
      
      if show_sp == 1 and i < start+10:
         plt.figure(min_bk_file,figsize=(8,6))
         plt.title("Frame "+str(i)+" - background image")
         plt.imshow(bk_fov_image,cmap=Colormap[color_sp])
         plt.colorbar()
         plt.waitforbuttonpress()
         plt.close()

         plt.figure(min_bk_file,figsize=(8,6))
         plt.title("Frame "+str(i)+" - neuron image")
         plt.imshow(neuron_fov_hole_img,cmap=Colormap[color_sp])
         plt.colorbar()
         plt.waitforbuttonpress()
         plt.close()

         plt.figure(min_bk_file,figsize=(8,6))
         plt.title("Frame "+str(i)+" - merged image")
         plt.imshow(bk_fov_neuron_hole_img,cmap=Colormap[color_sp])
         plt.colorbar()
         plt.waitforbuttonpress()
         plt.close()
      tif.write(bk_fov_neuron_hole_img,photometric='minisblack',contiguous=True)            
print("\n Complete!")

