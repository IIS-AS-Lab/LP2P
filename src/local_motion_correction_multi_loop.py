import cv2
from matplotlib import pyplot as plt
import time
import numpy as np
import sys,os
from skimage import io
import scipy.ndimage
from scipy import signal
import tifffile

from scipy import ndimage as ndi
from skimage.feature import peak_local_max

############## Keyboard event detection ############
p_version = "v3"
##########################################################   
argc_no = len(sys.argv)
if (argc_no < 2):
   print("Input file must be a bandpassed video")
   print("Output two files: 1. Motion corrected bandpassed video")
   print("                  2. Motion correction vectors") 
   print("No input!");
   print(" -i input file name  ");
   print("[Options]")
   print(" -o output file name ")
   print(" -it_no (1)          : the iterated number")
   print(" -random_ratio (0.1) : the ratio of random selected frame number for")
   print("                       calculating the global centroid")
   print(" -obs_range  (200)   : Diameter range for calcualting the centroid ")
   print(" -min_distance (20)  : Minimum distance between two peaks")
   print("             (40)    : For extreme motion cases")
   print(" -peak_mim   (5)     : Minimum intensity value of a peak")
   print(" -2p         (none)  : input video is two-photon ")
   print(" -mv_out     (none)  : output motion vector file ")
   print(" -obs2whole  (on/off): observation region to the whole ")
   print(" -avgOfdist  (none)  : calculate the average of matching peak distance ")
   print("                       default by the median")
   print(" -smooth     (none)  : smoothing all input farmes ")
   print(" -name_all   (none)  : show all parameters in the output file name")
   print(" -show    (all,off)  : all => show all detailed or analysis figures")
   print("                     : off => show nothing                         ")
   print(" -write_w_max(none)  : Write the maximum magnitude of corrected motions for each iteration")
   print(" -self_stop  (none)  :   Self-detection and auto stop (defaut: on)")
   print(" -self_stop_pdist (1): Set the maximun mv value if self_stop is on")
   print(" -rule_method (3)    : Select a method to initialize and adjust min_diatnce")
   print(" -pre_estimate_only (none)   : Calculate 75 percentile motion vector only ")
   print(" -pre_estimate_mv_out (none) : Output pre-calculated motion vectors")
   print(" -forced_stop_it_no  : Force to stop at the specific iteration")
   exit()
################# Set parameters ################
input_file = " "
output_file = " "

pre_estimate = 0
smoothing_sp = 0

mv_show_sp = 0 ##### Show mv list

w_max_no_sp = 0

self_stop = 0  ##### if self_stop = 1 => check the peak_max_dist => if <= self_stop_pdist, stop
self_stop_pdist = 3

forced_stop_itno = -1

##################################################################################
### Parameters for centroid calculation
##################################################################################
obs_range = 200

############################################################################################
### https://www.tutorialexample.com/understand-skimage-feature-peak_local_max-with-examples-python-tutorial/
### Parameters for the local maximum detection and matching searching
### function: peak_local_max(maxP_image, min_distance,threshold_rel,threshold_abs=None)
###           (a) min_distance: int, the minimal allowed distance separating peaks.
###                             To find the maximum number of peaks, use min_distance=1.
###           (b) threshold_rel: float, minimum intensity of peaks, 
###                              calculated as max(image) * threshold_rel
###           (c) threshold_abs: float, minimum intensity of peaks. By default,
###                              the absolute threshold is the minimum intensity of the image.
##############################################################################################
min_distance = 40         ## min_distance = peak_range
threshold_rel = 0.0       ## Check this part !!! threshold_rel = peak_min

random_ratio = 0.1  #### the ratio of the total frame number for calculating global centroid
random_threshold = 0.6   #### if the ratio > random_threshold, random_ratio = 1.0

obs2whole = 1             ## If the observation was calculated => obs2whole = 0
                          ## The observation is equal to the whole => obs2whole = 1
iteration_no = 10
##################################################################################
### Check the input parameters from argv
##################################################################################
rr_sp = 0; nssm_sp = 0; oR_sp = 0; pR_sp = 0; pM_sp = 0; name_all_sp = 0
Median_Avg_DoMP_sp = 1;

mv_out_sp = 0   ### if mv_out_sp = 1, output the motion vector file 
mv_out_all_sp = 0

pre_estimate_mv_out_sp = 0  ### if => 1, output the pre-calculated motion vector file 

c_method_no = 0 ## 0: Dynamic thresholding -> binarization -> calculate the centrod
                ##    of the region labelled with 1
                ## 1: Boundary reduction each iteration and calculate the centroid
                ##    of the reduced region

rule_method = 3 ## 0: Rule 0 (auto stop) : min_distance = 10*((int)(peak_max_dist/10) +1)
                ## 1: Rule 1 (3 it_nos)  :  min_distance = 10*((int)(peak_max_dist/10) +1)
                ##    Only running three iterations
                ## 2: Rule 2 : 
                ##    if L_flag == 1 and p < 2 :  min_distance = 10*((int)(peak_max_dist/10) +1)
                ##    else:  md_ratio = 1.5;   min_distance = (int)(md_ratio*peak_max_dist)
                ## 3: Rule 3 {md_ratio = 1.25} : min_distance = (int)(md_ratio*peak_max_dist)
version = 5

show_sp = 1 
Save_sp = 0
i=1

while (i<argc_no ):
   if (sys.argv[i] == "-i"):
      input_file = sys.argv[i+1]
      i += 2
   elif (sys.argv[i] == "-o"):
      output_file = sys.argv[i+1]
      i += 2
   elif (sys.argv[i] == "-it_no"):
      iteration_no = (int)((float)(sys.argv[i+1]))
      if iteration_no < 1 or iteration_no > 50:
         iteration_no = 1;   print("Assign the iteration number with a problem") 
      i += 2
   elif (sys.argv[i] == "-mv_out"):
      mv_out_sp = 1
      i += 1
   elif (sys.argv[i] == "-save"):
      Save_sp = 1
      i += 1
   elif (sys.argv[i] == "-mv_vsum_out"):
      mv_out_all_sp = 1
      i += 1
   elif (sys.argv[i] == "-2p"):
      obs2whole = 1
      i += 1
   elif (sys.argv[i] == "-obs2whole"):
      try:
         tsym = sys.argv[i+1]
         if tsym == "off":    obs2whole = 0;  i += 2
         elif tsym == "yes":  obs2whole = 1;  i += 2
         else:                i += 1
      except IndexError:
         i += 1
   elif (sys.argv[i] == "-random_ratio"):
      random_ratio = (float)(sys.argv[i+1]); rr_sp = 1
      i += 2
   elif (sys.argv[i] == "-obs_range"):
      obs_range = (int)(sys.argv[i+1]);      oR_sp = 1
      i += 2
   elif (sys.argv[i] == "-min_distance"):
      min_distance = (int)(sys.argv[i+1]);   pR_sp = 1
      i += 2
   elif (sys.argv[i] == "-peak_min"):
      threshold_rel = (float)(sys.argv[i+1]);pM_sp = 1
      i += 2
   elif (sys.argv[i] == "-avgOfdist"):
      Median_Avg_DoMP_sp = 0;
      i += 1
   elif (sys.argv[i] == "-smooth"): 
      smoothing_sp = 1;                      nssm_sp = 1
      i += 1
   elif (sys.argv[i] == "-name_all"): 
      name_all_sp = 1
      i += 1
   elif (sys.argv[i] == "-show"):
      try:
         tsym = sys.argv[i+1]
         if tsym == "all":
            show_sp = 2;  i += 2
         elif tsym == "off":
            show_sp = 0;  i += 2
         else:
            i += 1
      except IndexError:
         i += 1
   elif (sys.argv[i] == "-mv_show"):
      mv_show_sp = 1
      i += 1
   elif (sys.argv[i] == "-write_w_max"):
      w_max_no_sp = 1
      i += 1
   elif (sys.argv[i] == "-c_method"):
      c_method_no = (int)(sys.argv[i+1]);
      i += 2
   elif (sys.argv[i] == "-self_stop"):
      self_stop = 1
      i += 1
   elif (sys.argv[i] == "-self_stop_pdist"):
      self_stop_pdist = (int)(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-rule_method"):
      rule_method = (int)(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-forced_stop_it_no"):
      forced_stop_itno = (int)(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-pre_estimate_only"):
      pre_estimate = 1
      i += 1
   elif (sys.argv[i] == "-pre_estimate_mv_out"):
      pre_estimate_mv_out_sp = 1
      i += 1
   else:
      i += 2

if self_stop == 1:
   iteration_no = 20  ### maximum iteration no for self-detected termination

if rule_method == 4:
   if forced_stop_itno > iteration_no:
      iteration_no = forced_stop_itno
   else:
      iteration_no = 3
   md_ratio = 1
if rule_method == 5:
   if forced_stop_itno > iteration_no:
      iteration_no = forced_stop_itno
   else:
      iteration_no = 10
   md_ratio = 1

######## Check input parameters ###########
if (input_file == " "):
   print("No input")
   exit()
############ Detect Keyboard Event ###############
the_key = None
def press(event):
    global the_key
    the_key = event.key

###############################################################################
### Function for calculating various properties of two input images
### https://stackoverflow.com/questions/48888239/finding-the-center-of-mass-in-an-image
###############################################################################

from skimage import filters
from skimage.measure import regionprops
from skimage.measure import centroid
from skimage import measure

def Centroid_Calculation(Image,c_method_no):
   Image = Image.astype("int")
   if c_method_no == 0:
      threshold_value = filters.threshold_otsu(Image)
      labeled_foreground = (Image > threshold_value).astype(int)
      properties = regionprops(labeled_foreground, Image)
      Crow, Ccol = properties[0].centroid
      WCrow, WCcol = properties[0].weighted_centroid
      Imax =  properties[0].intensity_max
      Imin =  properties[0].intensity_min
   else:
      ####### Other method #####
      Crow, Ccol = centroid(Image)
      WCrow, WCcol = Crow, Ccol 
   
   return((Crow,Ccol,WCrow,WCcol))

##################################################

Length_boundary = 10
import random

if obs2whole == 1:
   random_ratio = 1.0

peak_max_dist = min_distance
P_max_list = []
min_dist_list = []
tm_p = 0

############################################################################################
if show_sp > 0: print(f"The expected iteration number = {iteration_no}")

for p in range(iteration_no):
   try:
      if p == 0:
         InFile = tifffile.TiffFile(input_file)
      else:
         InFile = tifffile.TiffFile(output_file)
   except IOError:
      if p == 0:
         print(f"\nInput file = \"{input_file}\" but Open it error! Quit!\n")
      else:
         print(f"\ntemporal file = \"{output_file}\" but Open it error! Quit!\n")
      quit()

   if p == 0:
      f_size = os.path.getsize(input_file) ; f_size = (f_size/1000000000)
      length = len(InFile.pages)
      (Row, Col) = InFile.pages[0].shape
      ROW = Row; COL= Col
      input_type = str(InFile.pages[0].dtype)
      if length < 10:
         print(f"{input_file} => not a video or too short! quit()")
         quit()
      ############################################################################################
      ###print("Set the calculation Boundary")
      ############################################################################################

      i_range = 5
      if p == 0:
         b_range = 80;   b_range_L = 80
      else:
         if c_method_no == 1:
            b_range_L = b_range;  b_range = 50 + (int)(p*i_range)

      if b_range > (int)(0.2*min(ROW,COL)):
         b_range = b_range_L
   
      R_start = b_range;  R_end = Row - b_range
      C_start = b_range;  C_end = Col - b_range

      star_no = 100;  star_step = (int)(length/star_no)

      if show_sp >0:  
         print(f"File size = {f_size:.2f} GB" )
         print(f"Shape of original video = {InFile.pages[0].shape}  Length = {length}")
         print(f" Original data type = ", input_type)
         print(f"[ R_start = {R_start}, C_start = {C_start} ] - [ R_end = {R_end}, C_end = {C_end}]")

   ############################################################################################
   ### G_centroid_array: Centroids calculated from the whole frame
   ############################################################################################
   ### Randomly select N frames only for speedup
   ############################################################################################

   if random_ratio < random_threshold:
      Rand_frame_no = (int)(random_ratio*length) 
      if show_sp > 0:
         print(f"Randomly select {Rand_frame_no} frames for calculating centroid")

      if p == 0:
         RC_list = []
      else:
         RC_list.clear()

      for i in range(Rand_frame_no):
         RC_list.append(random.randint(Length_boundary,length-Length_boundary))   

      r_star_step = 100
      r_star_no = (int)(Rand_frame_no/r_star_step)
      if show_sp > 0:
         for i in range(r_star_no):   print(".",end="",flush=True)
         for i in range(r_star_no):   print("\x08",end="",flush=True)
   
      if p == 0:    G_centroid_array = np.zeros((Rand_frame_no,6),dtype=int)
      else:         G_centroid_array[:,:]=0

      for i in range(Rand_frame_no):
         if show_sp > 0:
            if (i % r_star_step == 0):
               print("r",end="",flush=True)
         image = InFile.pages[i].asarray()
         if (smoothing_sp == 1):
            if not input_type.startswith('uint'):
               image = image.astype(np.int16)
            image = cv2.medianBlur(image,5)

         image = image.astype(np.int32)

         (gcR,gcC,gWcR,gWcC) = Centroid_Calculation(image[R_start:R_end,C_start:C_end],c_method_no)    
         G_centroid_array[i,0] = (int)(gcR)+b_range ;   G_centroid_array[i,1] = (int)(gcC)+b_range
         G_centroid_array[i,2] = (int)(gWcR)+b_range ;  G_centroid_array[i,3] = (int)(gWcC)+b_range
         """
         (gcR,gcC,gWcR,gWcC,gImax,gImin) = Centroid_Calculation(image[R_start:R_end,C_start:C_end]) 
         G_centroid_array[i,0] = (int)(gcR)+b_range ;   G_centroid_array[i,1] = (int)(gcC)+b_range
         G_centroid_array[i,2] = (int)(gWcR)+b_range ;  G_centroid_array[i,3] = (int)(gWcC)+b_range
         G_centroid_array[i,4] = (int)(gImax) ;         G_centroid_array[i,5] = (int)(gImin)   
         """
      if show_sp > 0: print("")
   else:
      ############################################################################################
      ### Calculate the gloabl unweighted nd weighted centroid of whole frame 
      ############################################################################################
      if show_sp > 0: print(f"Select all frames for calculating centroid")
      random_ratio = 1.0

      if show_sp > 0:
         for i in range(star_no): print(".",end="",flush=True)
         for i in range(star_no): print("\x08",end="",flush=True)

      if p == 0:  G_centroid_array = np.zeros((length,6),dtype=int)
      else:       G_centroid_array[:,:] = 0

      for i in range(length):
         if show_sp > 0: 
            if (i % star_step == 0):
               print("m",end="",flush=True)
         image = InFile.pages[i].asarray()
         if (smoothing_sp == 1):
            if not input_type.startswith('uint'):
               image = image.astype(np.int16)
            image = cv2.medianBlur(image,5)
         if (np.isnan(np.min(image[R_start:R_end,C_start:C_end]))):
            print(f"Appear NaN in the {i}th frame")
         (gcR,gcC,gWcR,gWcC) = Centroid_Calculation(image[R_start:R_end,C_start:C_end],c_method_no)
         G_centroid_array[i,0] = (int)(gcR)+b_range ;   G_centroid_array[i,1] = (int)(gcC)+b_range
         G_centroid_array[i,2] = (int)(gWcR)+b_range ;  G_centroid_array[i,3] = (int)(gWcC)+b_range

         """
         (gcR,gcC,gWcR,gWcC,gImax,gImin) = Centroid_Calculation(image[R_start:R_end,C_start:C_end])
         G_centroid_array[i,0] = (int)(gcR)+b_range ;   G_centroid_array[i,1] = (int)(gcC)+b_range
         G_centroid_array[i,2] = (int)(gWcR)+b_range ;  G_centroid_array[i,3] = (int)(gWcC)+b_range
         G_centroid_array[i,4] = (int)(gImax) ;         G_centroid_array[i,5] = (int)(gImin)
         """
      if show_sp > 0:   print("")

   ############################################################################################
   ### Check the inter-lens shift by variance
   ############################################################################################
   if show_sp > 0: print("Calculate the mean of centroid and the standard deviation")

   gR_avg  = (int)(np.mean(G_centroid_array[:,0])); gC_avg  = (int)(np.mean(G_centroid_array[:,1])) 
   gWR_avg = (int)(np.mean(G_centroid_array[:,2])); gWC_avg = (int)(np.mean(G_centroid_array[:,3]))

   gR_std  = np.std(G_centroid_array[:,0]);  gC_std  = np.std(G_centroid_array[:,1]) 
   gWR_std = np.std(G_centroid_array[:,2]);  gWC_std = np.std(G_centroid_array[:,3]) 
   """
   print(f"(gR_avg,gC_avg)=({gR_avg},{gC_avg})   (gWR_avg,gWC_avg)=({gWR_avg},{gWC_avg})")
   print(f"(gR_std,gC_std)=({gR_std},{gC_std})   (gWR_std,gWC_std)=({gWR_std},{gWC_std})")
   """
   Ref_Cen_R = gWR_avg
   Ref_Cen_C = gWC_avg

   """
   Ref_Cen_R = gR_avg
   Ref_Cen_C = gC_avg
   """
   if show_sp > 0:   print(f"Reference centroid  => ({Ref_Cen_R},{Ref_Cen_C})")

   #############################################################################################
   ### Finetune Motion Correction for the selected region
   ###       centered at (Ref_Cen_R,Ref_Cen_C) with a fixed range (Obs_Range)
   ###    (Cen_row_min,Cen_col_min)----------------------------
   ###                |                   |<.....Obs_Range...>|
   ###                |             (Cen_R,Cen_C)             |
   ###                |                                       |
   ###                -----------------------------(Cen_row_max,Cen_col_max)
   ### Obs_Range: the half range of observed centroid, and can be different from obs_region
   #############################################################################################

   Obs_Range = obs_range

   mm_range = (int)(min(COL/2,ROW/2))
   if Obs_Range > mm_range or  obs2whole == 0:
      Obs_Range = (int)(min(COL/2,ROW/2))

   (Cen_R,Cen_C) = (Ref_Cen_R,Ref_Cen_C)   ### Centroid of reference image
   Cen_row_min = max(Cen_R - Obs_Range,0)  ; Cen_col_min = max(Cen_C - Obs_Range,0)
   Cen_row_max = min(Cen_R + Obs_Range,Row); Cen_col_max = min(Cen_C + Obs_Range,Col)
   if show_sp > 0:
      print(f"Calcation region: Obs_Range = {Obs_Range} obs2whole = {obs2whole}")
      print("")
      print(f"({Cen_row_min:04d},{Cen_col_min:04d})---------------------------------------------|")
      print(f"     |                                                  |")
      print(f"     |                   ({Cen_R:04d},{Cen_C:04d})                    |")
      print(f"     |<--------- {Obs_Range:04d} ------->|                         |")
      print(f"     |---------------------------------------------({Cen_row_max:04d},{Cen_col_max:04d})")
      print("")

   ############################################################################################
   ### Lmax_points : All local peaks for each frame (in a list format)
   ### maxP_image  : local peak distribution
   ### Pimg        : the selected region array
   ############################################################################################

   if p == 0:
      Lmax_points = {}
      maxP_image = np.zeros((Row,Col),dtype=int)
      Pimg = np.zeros((Cen_row_max-Cen_row_min,Cen_col_max-Cen_col_min),dtype=int)
   else:
      ###
      Lmax_points.clear();  maxP_image[:,:] = 0 ;   Pimg[:,:] = 0

   peak_range_auto = 1
   if p == 0 and rule_method < 4:
      if show_sp > 0:
         print(f"Rule Method {rule_method} ",end="")
         print(f"Pre-calculate the initial min_distance")
         for i in range(star_no):   print(".",end="",flush=True)
         for i in range(star_no):   print("\x08",end="",flush=True)

      t_maxP_image = np.zeros((Row,Col),dtype=int)
      t_Pimg = np.zeros((Cen_row_max-Cen_row_min,Cen_col_max-Cen_col_min),dtype=int)
      t_Lmax_points = {}

      for i in range(length):
         t_Lmax_points[i] = []
         if show_sp >0:
            if (i % star_step == 0):
               print(">",end="",flush=True)
         image = InFile.pages[i].asarray()
         """
         image = image.astype(np.int16)
         """
         if (smoothing_sp == 1 and p ==0):
            image = cv2.medianBlur(image,5)
         image = image.astype(np.int32)

         t_Pimg = image[Cen_row_min:Cen_row_max,Cen_col_min:Cen_col_max]

         ###coord = peak_local_max(t_Pimg,min_distance=min_distance,threshold_rel=threshold_rel)
         ###coord = peak_local_max(t_Pimg,min_distance,threshold_rel)
         coord = peak_local_max(t_Pimg,min_distance=min_distance)

         coord[:, 0] += Cen_row_min
         coord[:, 1] += Cen_col_min
         t_maxP_image[coord[:, 0],coord[:, 1]] += 1
         t_Lmax_points[i].append((coord[:, 0],coord[:, 1]))

         if i == 2 or i == (int)(length/2):
            if Save_sp >0:
               from matplotlib.patches import Rectangle

               image = InFile.pages[i].asarray()
               plt.figure(input_file,figsize=(8,6))
               pos_str = "("+str(Cen_row_min)+","+str(Cen_col_min)+")-"
               pos_str +="("+str(Cen_row_max)+","+str(Cen_col_max)+")"
               plt.title(input_file+": "+str(i)+" frame "+pos_str)
               plt.imshow(image,cmap=plt.cm.hot)
               row_len = Cen_row_max - Cen_row_min
               col_len = Cen_col_max - Cen_col_min
               plt.gca().add_patch(Rectangle((Cen_col_min,Cen_row_min),col_len,row_len,
                    edgecolor='white',
                    facecolor='none',
                    lw=2)) 
               c_row = (int)((Cen_row_max+Cen_row_min)/2)
               c_col = (int)((Cen_col_max+Cen_col_min)/2)           
               plt.plot(c_col,c_row,'wo')
               plt.tight_layout()
               plt.waitforbuttonpress()
               save_file = input_file+"."+str(i)+"_ROI.png"
               plt.savefig(save_file)
               plt.close()

      if show_sp >0: print("")

      t_max_peak_peak_no = np.max(t_maxP_image)
      t_local_peak_no = np.count_nonzero(t_maxP_image)
      t_local_peak_no_ratio = t_local_peak_no / (ROW*COL)

      ###t_GlmPs = peak_local_max(t_maxP_image,min_distance=min_distance,threshold_rel=threshold_rel)
      ###t_GlmPs = peak_local_max(t_maxP_image,min_distance,threshold_rel)
      t_GlmPs = peak_local_max(t_maxP_image,min_distance=min_distance)

      t_anchor_peak_no = len(t_GlmPs)

      psum =0
      for k in range(len(t_GlmPs)):
         psum += t_maxP_image[t_GlmPs[k,0],t_GlmPs[k,1]]
         if show_sp >0: 
            if k < 10:
               print(f" {t_maxP_image[t_GlmPs[k,0],t_GlmPs[k,1]]},",end="")
      if show_sp >0: 
         print("")
         print(f"Anaysis the pre-processing result") 
         print(f"Input video : {input_file}")
         print(f"min_distance = {min_distance}")
         print(f"t_peak_no_sum = {np.sum(t_maxP_image)}")
         print(f"t_max_peak_peak_no = {t_max_peak_peak_no}   ",end="")
         print(f"t_max_peak_no/all = {t_max_peak_peak_no/t_local_peak_no}")
         print(f"t_local_peak_no = {t_local_peak_no}  ",end="")
         print(f"t_local_peak_no_ratio = {t_local_peak_no_ratio}")
         print(f"t_anchor_peak_no = {t_anchor_peak_no}")
         print(f"The total local peak numbers in anchors Psum = {psum} Psum/anchor_no = {psum/len(t_GlmPs)}")

      ########################################################################
      ### high peak_maximum, short peak range  => small motion effect
      ### low peak_maximum, long peak range    => high motio effect
      ########################################################################

      t_MAXpoints = []
      for i in range(len(t_GlmPs)):
         if (t_GlmPs[i, 0] < Cen_row_max and t_GlmPs[i, 0] > Cen_row_min):
            if (t_GlmPs[i, 1] < Cen_col_max and t_GlmPs[i, 1] > Cen_col_min):
               t_MAXpoints.append([t_GlmPs[i, 0],t_GlmPs[i, 1]])
      t_Peaks = np.zeros((len(t_MAXpoints),2),dtype=int)
      for i in range(len(t_MAXpoints)):
         [t_Peaks[i,0],t_Peaks[i,1]] = t_MAXpoints[i]

      t_dLV_0 = {};    t_dLV_1 ={};  t_dLMeanV = []

      for i in range(length):
         t_dLV_0[i] = [];    t_dLV_1[i] = []
         (Lps_0,Lps_1) = t_Lmax_points[i][0]
         for j in range(len(Lps_0)):
            dP = (t_Peaks[:,0]-Lps_0[j])**2 +(t_Peaks[:,1]-Lps_1[j])**2
            idx_min = np.argmin(dP)

            t_dLV_0[i].append(Lps_0[j]-t_Peaks[idx_min,0])
            t_dLV_1[i].append(Lps_1[j]-t_Peaks[idx_min,1])

         if Median_Avg_DoMP_sp == 1:
            Avg_0 = np.median(np.asarray(t_dLV_0[i]))
            Avg_1 = np.median(np.asarray(t_dLV_1[i]))
         else:
            Avg_0 = np.mean(np.asarray(t_dLV_0[i]))
            Avg_1 = np.mean(np.asarray(t_dLV_1[i]))

         t_dLMeanV.append((Avg_0,Avg_1))

      ay_dLMeanV = abs(np.asarray(t_dLMeanV))

      if pre_estimate_mv_out_sp == 1:
         if (output_file == " "):
            if input_file[-5:] == ".tiff":
               pemv_file = input_file[:-5]
            else:
               pemv_file = input_file
         pemv_file += ".PE.mv.txt" 
         print(f"Open the pre-estimated motion file {pemv_file}")
         print("Writing")
         with open(pemv_file,'w') as f:     
            for i in range(length):
               (A_0,A_1) = t_dLMeanV[i]
               dR_row = (int)(A_0+0.5);        dR_col = (int)(A_1+0.5)
               CPBox_row = Row - abs(dR_row); CPBox_col = Col - abs(dR_col)

               target_start_r = max(dR_row,0);    target_start_c = max(dR_col,0)
               target_end_r = target_start_r + CPBox_row ; target_end_c = target_start_c + CPBox_col

               source_start_r = abs(min(dR_row,0));  source_start_c = abs(min(dR_col,0))
               source_end_r = source_start_r + CPBox_row;   source_end_c = source_start_c + CPBox_col

               reg_Lr = target_start_r; reg_Lc = target_start_c; reg_Rr = target_end_r; reg_Rc = target_end_c
               trg_Lr = source_start_r; trg_Lc = source_start_c; trg_Rr = source_end_r; trg_Rc = source_end_c

               f.write(str(dR_row)+","+str(dR_col)+",")
               f.write(str(trg_Lr)+","+str(trg_Rr)+","+str(trg_Lc)+","+str(trg_Rc)+",")
               f.write(str(reg_Lr)+","+str(reg_Rr)+","+str(reg_Lc)+","+str(reg_Rc)+"\n")            

         print("Complete!")

      """
      pkmean_0_90 = np.percentile(ay_dLMeanV[:,0],90)
      pkmean_1_90 = np.percentile(ay_dLMeanV[:,1],90)
      """
      per = 75
      pkmean_0 = np.percentile(ay_dLMeanV[:,0],per)
      pkmean_1 = np.percentile(ay_dLMeanV[:,1],per)

      if show_sp > 0:
         print(f"Percentile {per} vector = ({pkmean_0},{pkmean_1})")
      if pre_estimate == 1 :
         print(f"{input_file} {per} = ({pkmean_0},{pkmean_1})")
         quit()

      pkmax = (int)(np.max(t_dLMeanV))
      pkmin = (int)(abs(np.min(t_dLMeanV)))

      peak_max_dist = max(pkmax,pkmin)

      md_ratio = 1.5
      if (peak_max_dist > 10):
         min_distance = 40
         L_flag  = 1
      else:
         L_flag  = 0      
         min_distance = (int)(md_ratio*peak_max_dist)

      if show_sp > 0:
         print(f"Pre-calculated 75 percentile vector = ({pkmean_0},{pkmean_1})") 
         print(f"Pre-calculated peak_max_dist = {peak_max_dist} and md_ratio = {md_ratio}",end="")
         print(f"Reset initial min_distance = {min_distance}")

   if show_sp >0:
      if rule_method >3:
          print(f"Rule method {rule_method}: Preset iteration no = {iteration_no}")
   print(f"{p}th iteration processing min_distance = {min_distance}")

   min_dist_list.append(min_distance)

   ############################################################################################# 
   if show_sp > 0:
      for i in range(star_no):  print(".",end="",flush=True)
      for i in range(star_no):  print("\x08",end="",flush=True)

   for i in range(length):
      Lmax_points[i] = []
      if show_sp > 0:
         if (i % star_step == 0):  print(">",end="",flush=True)
      ############################################################################################
      ### Only conisder the region centered at refnce centroid wirth a fixed range
      ############################################################################################
      image = InFile.pages[i].asarray()
      """
      image = image.astype(np.int16)
      """
      if (smoothing_sp == 1):
         image = cv2.medianBlur(image,5)

      image = image.astype(np.int32)
      Pimg = image[Cen_row_min:Cen_row_max,Cen_col_min:Cen_col_max]

      ###coordinates = peak_local_max(Pimg,min_distance=min_distance,threshold_rel=threshold_rel)
      ###coordinates = peak_local_max(Pimg,min_distance,threshold_rel)
      coordinates = peak_local_max(Pimg,min_distance=min_distance)

      coordinates[:, 0] += Cen_row_min
      coordinates[:, 1] += Cen_col_min
      maxP_image[coordinates[:, 0],coordinates[:, 1]] += 1 
      Lmax_points[i].append((coordinates[:, 0],coordinates[:, 1]))  
   if show_sp > 0: 
      ##print(f"coordinates = peak_local_max(Pimg,{min_distance},{threshold_rel})")
      print(f"\ncoordinates = peak_local_max(Pimg,{min_distance})")
   maxall = np.max(maxP_image)

   if p == 0:
      Max_Anchor_Peak_NO = []
   Max_Anchor_Peak_NO.append(maxall)
   if show_sp > 0: 
      print(f"The maximum number in the map maxP_image = {np.max(maxP_image)}")
   
   show_MPimage = np.copy(maxP_image)

   if show_sp > 1:
      plt.figure(input_file,figsize=(8,6))
      plt.title(input_file+" maxP_image")
      plt.imshow(maxP_image,cmap=plt.cm.Greys)
      plt.waitforbuttonpress()
      plt.close()

      plt.figure(input_file,figsize=(8,6))
      plt.title(input_file+" show_MPimage ")
      plt.imshow(show_MPimage,cmap=plt.cm.hot)
      plt.waitforbuttonpress()
      plt.close()
     
      plt.figure(input_file,figsize=(8,6))
      plt.title(input_file+" maxP_image[351-344]")
      plt.plot(maxP_image[344])
      plt.plot(maxP_image[345]+20)
      plt.plot(maxP_image[346]+40)
      plt.plot(maxP_image[347]+60)
      plt.plot(maxP_image[348]+80)
      plt.plot(maxP_image[349]+100)
      plt.plot(maxP_image[350]+120)
      plt.plot(maxP_image[351]+140)
      plt.waitforbuttonpress()
      plt.close()
      
   ########################################################################################
   ### Global Maximum Peaks (GLMP)
   ########################################################################################

   GlmPs = peak_local_max(maxP_image, min_distance=min_distance)

   if show_sp > 0:
      print(f"GlmPs = peak_local_max(maxP_image,{min_distance})")
      print(f"The anchor number detected in maxP_image = {len(GlmPs)} ")

   if p == 0:
      anchor_peak_no = {}
   anchor_peak_no[p] = [] 

   psum =0 
   for k in range(len(GlmPs)):
      ###print(f"{k} : maxP_image({GlmPs[k,0]},{GlmPs[k,1]}) = {maxP_image[GlmPs[k,0],GlmPs[k,1]]}")
      psum += maxP_image[GlmPs[k,0],GlmPs[k,1]]
      if k < 10:
         anchor_peak_no[p].append(maxP_image[GlmPs[k,0],GlmPs[k,1]])
         if show_sp > 0:
            print(f" {maxP_image[GlmPs[k,0],GlmPs[k,1]]},",end="")
   if show_sp > 0:
      print("")
      print(f"The total local peak numbers in anchors Psum = {psum} Psum/anchor_no = {psum/len(GlmPs)}") 

   if show_sp > 1:
      rad = 6
      for i in range(len(GlmPs)):
         [x,y] = [GlmPs[i, 0],GlmPs[i, 1]]
         for r in range(-1*rad,rad+1):
            for c in range(-1*rad,rad+1):
               if (r*r+c*c < rad*rad):
                  if x+r >-1 and x+r < ROW and y+c > -1 and y+c < COL:
                     show_MPimage[x+r,y+c] = 128     
      plt.figure(input_file,figsize=(8,6))
      plt.title(input_file+" show_MPimage + GlmPs")
      plt.imshow(show_MPimage,cmap=plt.cm.hot)
      plt.waitforbuttonpress()
      plt.close()

   ##########################################################################################
   ### Filter out the max points not in the Centroid region
   ### return from peak_local_max: (row, column, …) coordinates of peaks.
   ###   *** MAXpoints: Selected local peaks in a list format)
   ###   *** Peaks: Selected peaks in an array format 
   ##########################################################################################
   MAXpoints = []
   for i in range(len(GlmPs)):
      if (GlmPs[i, 0] < Cen_row_max and GlmPs[i, 0] > Cen_row_min): 
         if (GlmPs[i, 1] < Cen_col_max and GlmPs[i, 1] > Cen_col_min):
            MAXpoints.append([GlmPs[i, 0],GlmPs[i, 1]])
   Peaks = np.zeros((len(MAXpoints),2),dtype=int)
   for i in range(len(MAXpoints)):
      [Peaks[i,0],Peaks[i,1]] = MAXpoints[i]
   ##########################################################################################

   if show_sp > 1:
      plt.figure("Local Peaks",figsize=(8,6))
      plt.title("Detected and selected peaks")
      ###plt.imshow(maxP_image)
      plt.imshow(show_MPimage)
      plt.plot(GlmPs[:, 1], GlmPs[:, 0], 'w.')
      plt.plot(Peaks[:, 1], Peaks[:, 0], 'r.')
      plt.plot(Cen_C,Cen_R,'wo')
      plt.waitforbuttonpress()
      plt.close()

   #############################################################################################
   ### Calculation of the matching distance between the maximum points
   #############################################################################################
   ### print("Motion correction")

   #############################################################################################
   ### dLV:  Difference vectors of all local peaks to global max point
   ### Mean: The mean vector of difference vectors in dLV for each frame
   ### dLMeanV: the shift vectors
   #############################################################################################
   dLV_0 = {};    dLV_1 ={}
   dLMeanV = []

   #############################################################################################

   if show_sp > 0:
      print("Find the ",end="")
      if Median_Avg_DoMP_sp == 1:  print("median ",end="")
      else:   print("average ",end="")
      print("of the distances between local to reference peaks")

   if show_sp > 0:
      for i in range(star_no):  print(".",end="",flush=True)
      for i in range(star_no):  print("\x08",end="",flush=True)

   for i in range(length):  
      dLV_0[i] = [];    dLV_1[i] = []
      if show_sp > 0:
         if (i % star_step == 0):   print("@",end="",flush=True)

      (Lps_0,Lps_1) = Lmax_points[i][0]
      for j in range(len(Lps_0)):
         dP = (Peaks[:,0]-Lps_0[j])**2 +(Peaks[:,1]-Lps_1[j])**2
         idx_min = np.argmin(dP)
         dLV_0[i].append(Lps_0[j]-Peaks[idx_min,0])
         dLV_1[i].append(Lps_1[j]-Peaks[idx_min,1])

      if Median_Avg_DoMP_sp == 1:
         Avg_0 = np.median(np.asarray(dLV_0[i]))
         Avg_1 = np.median(np.asarray(dLV_1[i]))
      else:
         Avg_0 = np.mean(np.asarray(dLV_0[i]))
         Avg_1 = np.mean(np.asarray(dLV_1[i]))

      dLMeanV.append((Avg_0,Avg_1))
   
      if show_sp > 1:
         print(f"{i} peak no. = {len(dLV_0[i])}       (Avg_0,Avg_1) = ({Avg_0},{Avg_1})")
         """
         print(f"(Avg_0,Avg_1) = ({Avg_0},{Avg_1}) ")
         print(f"np.asarray(dLV_1[i]) = {np.asarray(dLV_1[i])}")
         for b in range(len(dLV_0[i])):
            print(f"({dLV_0[i][b]},{dLV_1[i][b]}) ",end="")
         print("") 
         """
         plt.figure("Local Peaks",figsize=(8,6))
         p_str = input_file+"\nFrame="+str(i)
         p_str += " dv = ("+str(f'{Avg_0:.1f}')+","+str(f'{Avg_1:.1f}')+")"
         plt.title(p_str)
         image = InFile.pages[i].asarray()
         ###plt.imshow(image)
         plt.imshow(show_MPimage)
         plt.plot(Lps_1[:]+Avg_1, Lps_0+Avg_0, 'wo')    ### Adjusted local max peaks
         ##plt.plot(Peaks[:, 1], Peaks[:, 0], 'r.')   ### Max of max peaks
         plt.legend(['local peaks','Anchors'])
         plt.waitforbuttonpress()
         plt.close()
   
         plt.figure("Local Peaks",figsize=(8,6))
         plt.title(p_str)
         rdV_0 = np.asarray(dLV_0[i]); rdV_1 = np.asarray(dLV_1[i])
         plt.plot(rdV_1[:], rdV_0[:], 'r.')
         plt.plot(Avg_1,Avg_0,'ro')
         plt.axvline(x= 0,color='b')
         plt.axhline(y= 0,color='b')
         plt.xlabel("diff-col")
         plt.ylabel("diff-row")
         plt.xlim([-20,20])
         plt.ylim([-20,20])
         plt.waitforbuttonpress()
         plt.close()
      
         if i > 15:
            show_sp = 2
   if show_sp > 0:
      print("")
 
   pkmax = (int)(np.max(dLMeanV))
   pkmin = (int)(abs(np.min(dLMeanV)))

   peak_max_dist = max(pkmax,pkmin)
   P_max_list.append(peak_max_dist)

   if show_sp > 0: 
      if rule_method < 4:
         print(f"\nL_flag = {L_flag} ",end="")
      print(f"peak_max_dist = {peak_max_dist} ",end="") 

   if rule_method == 0:
      if show_sp > 0: print(" Rule 0 (auto stop) ",end="")
      min_distance = 10*((int)(peak_max_dist/10) +1)

   if rule_method == 1:
      if show_sp > 0: print(" Rule 1 (3 iterations) ",end="")
      min_distance = 10*((int)(peak_max_dist/10) +1) 
      if p == 2:
         self_stop = 1; tm_p = p+1 

   if rule_method == 2: 
      if show_sp > 0: print(" Rule 2 ",end="")
      if L_flag == 1 and p < 3 :
         min_distance = 10*((int)(peak_max_dist/10) +1)
      else:
         md_ratio = 1.25 
         min_distance = (int)(md_ratio*peak_max_dist)
   
   if rule_method == 3: 
      md_ratio = 1.25
      if show_sp > 0: print(f" Rule 3 {md_ratio} ",end="")
      min_distance = (int)(md_ratio*peak_max_dist)
   ####################################################

   if rule_method > 3:
      if show_sp > 0: print(f" Rule {rule_method} ",end="")
      min_distance = 10*((int)(peak_max_dist/10) +1)
   
   """
   md_ratio = np.power(0.8,p+1)+1 
   min_distance = (int)(md_ratio*peak_max_dist)
   """
   if show_sp > 0: 
      print(f" The min_distance for the next iteration = {min_distance} ( md_ratio={md_ratio}*peak_max_dist)")      
   ### Termination detection
   if peak_max_dist < self_stop_pdist:
      self_stop = 1; tm_p = p+1

   if self_stop == 1:
      if show_sp > 0: 
         print(f"Check self_stop = {self_stop} peak_max_dist = {peak_max_dist} ",end="")
         print(f"self_stop_pdist = {self_stop_pdist}")
      if peak_max_dist <= self_stop_pdist:
         print(f"{p+1}th iteration: peak_max_dist <= self_stop_pdist {self_stop_pdist} ")
         print("Preparing to terminate the correction")
         tm_p = p+1

   ###########################################################################
   ### Avg_Vect: The final shift vectors in an array format
   ###########################################################################
   if p == 0:
      Avg_Vect_all = np.zeros((length,2),dtype=int)      
      MV_List = {}
      for k in range(iteration_no):
         MV_List[k] = []

   Avg_Vect = np.zeros((length,2),dtype=int)
   print(f"p = {p} iteration_no = {iteration_no}")
   for i in range(len(dLMeanV)):
      (A_0,A_1) = dLMeanV[i]
      ##Avg_Vect[i,0] = (int)(A_0+0.5);        Avg_Vect[i,1] = (int)(A_1+0.5)
      Avg_Vect[i,0] = (int)(A_0);        Avg_Vect[i,1] = (int)(A_1)
      Avg_Vect_all[i,0] += Avg_Vect[i,0];    Avg_Vect_all[i,1] += Avg_Vect[i,1]
      MV_List[p].append((Avg_Vect[i,0],Avg_Vect[i,1]))
      """
      if show_sp == 1:
         if i < 100:
            print(f"{i}: ({Avg_Vect[i,0]},{Avg_Vect[i,0]})")
      """
   if show_sp > 1: 
      plt.figure("Correlated Shift Vectors",figsize=(12,6))
      plt.plot(Avg_Vect[:,0],'b-')
      plt.plot(Avg_Vect[:,1],'k-')

      plt.waitforbuttonpress()
      plt.close()

   if p == 0:
      #####################################################################################
      ### Parameters:
      ### (smooth,sampling_ratio,obs_range,min_distance,threshold_rel,p_version)
      ### ( ns/sm,rd            ,oR       ,pR          ,pM           ,v?       )
      ### peak_range = min_distance,  peak_min = threshold_rel  
      #####################################################################################
      print("Parameters:")
      print(f"   sampling_ratio = {random_ratio}   obs_range = {obs_range} ")
      print(f"   min_distance = {min_distance}   threshold_rel = {threshold_rel} ") 
      print(f"   p_version = {p_version}")

      if (output_file == " "):   #### Only execute in the 1st iteration #####
         if input_file[-5:] == ".tiff":
            output_file = input_file[:-5]
         else:
            output_file = input_file
         if iteration_no > 1:
            output_file += "."+str(iteration_no)
            ###output_file += 'LMCs.'
            output_file += 'LMC.'
         else:
            output_file += '.1LMC.'

         ##################################################################################
         ###
         ##################################################################################

         if rr_sp == 1 or name_all_sp == 1 :
            rR = (int)(random_ratio * 10.)
            if rR == 10: 
               output_file += "all."  ### all frames are selected
            else:
               output_file += "rd0"+str(rR)+"."  ### randomly selected frames
         if nssm_sp == 1 or name_all_sp == 1 :
            if smoothing_sp == 1:
               output_file += "sm."
            else:
               output_file += "ns."
         if oR_sp == 1 or name_all_sp == 1 :
            output_file += "oR"+str(obs_range)+"."
         if pR_sp == 1 or name_all_sp == 1 :
            output_file += "pR"+str(min_distance)+"."
         if pM_sp == 1 or name_all_sp == 1 :
            output_file += "pM"+str(threshold_rel)+"."
         if Median_Avg_DoMP_sp == 0 or name_all_sp == 1 :
            if Median_Avg_DoMP_sp == 1:
               output_file += "MedDis."
            else:
               output_file += "AvgDis."   

         output_file += "r"+str(rule_method)+"."
         output_file += "s"+str(self_stop_pdist)+"."
         output_file += "v"+str(version)+"."
         output_file += "tiff"

      print("output_file = ",output_file)

   mv_file = output_file+".v"+str(version)+"."+str(p+1)+"Lmv.txt"
   ### print("motion vector txt file = ",mv_file)
   
   if os.path.exists(output_file):
      os.remove(output_file)
      if show_sp > 0:
         print(f"Delete {output_file} and Create a new one!")

   ###########################################################################

   if (len(dLMeanV) != length):
      print("The total mv number is incorrect! quit")
      quit()

   if mv_out_sp == 1:
      print("Writing the motion vectors to ",mv_file)
      with open(mv_file,'w') as f:
         f.write("###########################################################################\n")
         f.write("# Avg_Vect[i,0],Avg_Vect[i,1],trg_Lr,trg_Rr,trg_Lc,trg_Rc,reg_Lr,reg_Rr,reg_Lc,reg_Rc\n")
         f.write("# dR_row = Avg_Vect[i,0];              dR_col = Avg_Vect[i,1]              \n")
         f.write("# CPBox_row = Row - abs(dR_row);       CPBox_col = Col - abs(dR_col)       \n")
         f.write("# reg_Lr = max(dR_row,0);              reg_Lc = max(dR_col,0)              \n")
         f.write("# reg_Rr = target_start_r + CPBox_row; reg_Rc = target_start_c + CPBox_col \n")
         f.write("# trg_Lr = abs(min(dR_row,0));         trg_Lc = abs(min(dR_col,0))         \n")
         f.write("# trg_Rr = source_start_r + CPBox_row; trg_Rc = source_start_c + CPBox_col \n")
         f.write("# Img[:,:] = 0.0;                                                          \n")
         f.write("# Img[trg_Lr:trg_Rr,trg_Lc:trg_Rc] = images[i][reg_Lr:reg_Rr,reg_Lc:reg_Rc]\n")
         f.write("# images[i][:,:] = Img[:,:]                                                \n")
         f.write("###########################################################################\n")
         f.write("*,"+str(gWcR)+","+str(gWcC)+","+str(Cen_row_min)+","+str(Cen_col_min)+","+str(Cen_row_max)+","+str(Cen_col_max)+"\n")

         for i in range(length):

            ##dR_row = -1*Avg_Vect[i,0];    dR_col = -1*Avg_Vect[i,1]   ## for Peaks[idx_min,1]-Lps_1[j])
            dR_row = Avg_Vect[i,0];    dR_col = Avg_Vect[i,1]           ## for Lps_1[j] - Peaks[idx_min,1]

            CPBox_row = Row - abs(dR_row)        ### CPBox       : Corresponding Block Size
            CPBox_col = Col - abs(dR_col)        ###

            target_start_r = max(dR_row,0);    target_start_c = max(dR_col,0)
            target_end_r = target_start_r + CPBox_row ; target_end_c = target_start_c + CPBox_col

            source_start_r = abs(min(dR_row,0));  source_start_c = abs(min(dR_col,0))
            source_end_r = source_start_r + CPBox_row;   source_end_c = source_start_c + CPBox_col

            reg_Lr = target_start_r; reg_Lc = target_start_c; reg_Rr = target_end_r; reg_Rc = target_end_c
            trg_Lr = source_start_r; trg_Lc = source_start_c; trg_Rr = source_end_r; trg_Rc = source_end_c

            f.write(str(Avg_Vect[i,0])+","+str(Avg_Vect[i,1])+",")
            f.write(str(trg_Lr)+","+str(trg_Rr)+","+str(trg_Lc)+","+str(trg_Rc)+",")
            f.write(str(reg_Lr)+","+str(reg_Rr)+","+str(reg_Lc)+","+str(reg_Rc)+"\n") 
   if show_sp > 0:  
      print("Compelete!")

   if p == 0:
      Img = np.zeros((ROW,COL),dtype=float)

   if show_sp > 0:
      for i in range(star_no):  print(".",end="",flush=True)
      for i in range(star_no):  print("\x08",end="",flush=True)

   with tifffile.TiffWriter(output_file,bigtiff=True) as tif:
      for i in range(length):
         if show_sp > 0:
            if (i % star_step == 0):  print("x",end="",flush=True)

         dR_row = Avg_Vect[i,0];    dR_col = Avg_Vect[i,1]           ## for Lps_1[j] - Peaks[idx_min,1]

         CPBox_row = Row - abs(dR_row)        ### CPBox       : Corresponding Block Size
         CPBox_col = Col - abs(dR_col)        ###
   
         target_start_r = max(dR_row,0);    target_start_c = max(dR_col,0)
         target_end_r = target_start_r + CPBox_row ; target_end_c = target_start_c + CPBox_col

         source_start_r = abs(min(dR_row,0));  source_start_c = abs(min(dR_col,0))
         source_end_r = source_start_r + CPBox_row;   source_end_c = source_start_c + CPBox_col

         reg_Lr = target_start_r; reg_Lc = target_start_c; reg_Rr = target_end_r; reg_Rc = target_end_c
         trg_Lr = source_start_r; trg_Lc = source_start_c; trg_Rr = source_end_r; trg_Rc = source_end_c
   
         image = InFile.pages[i].asarray()
         Img[:,:] = 0.0
         ##Img[:,:] = image[:,:]
         Img[trg_Lr:trg_Rr,trg_Lc:trg_Rc] = image[reg_Lr:reg_Rr,reg_Lc:reg_Rc] 
         ##image[:,:] = 0.0
         image[:,:] = Img[:,:]
         ##image = image.astype('e')
         tif.write(image,photometric='minisblack',contiguous=True)

   if show_sp > 0:
      print("")
      print(f"it_no = {p}  self_stop = {self_stop}  tm_p = {tm_p} ")

   if self_stop == 1 and tm_p > 0:
      break

   if forced_stop_itno == p+1:    ### Forced stop at the pth iteration
      self_stop = 1; tm_p = p+1
      break

if show_sp > 0:
   print("No of the local peak numbers of max anchor")
   for i in range(len(Max_Anchor_Peak_NO)):
      print(f"{Max_Anchor_Peak_NO[i]}, ",end="")
   print("")

it_sp = iteration_no

if mv_out_all_sp == 1:
   mv_file_all = input_file + ".v"+str(version)+".Lmv.vsum" 
   print("Writing the motion vectors to ",mv_file)
   with open(mv_file_all,'w') as f_all:
      f_all.write("###########################################################################\n")
      f_all.write("# Avg_Vect[i,0],Avg_Vect[i,1],trg_Lr,trg_Rr,trg_Lc,trg_Rc,reg_Lr,reg_Rr,reg_Lc,reg_Rc\n")
      f_all.write("# dR_row = Avg_Vect[i,0];              dR_col = Avg_Vect[i,1]              \n")
      f_all.write("# CPBox_row = Row - abs(dR_row);       CPBox_col = Col - abs(dR_col)       \n")
      f_all.write("# reg_Lr = max(dR_row,0);              reg_Lc = max(dR_col,0)              \n")
      f_all.write("# reg_Rr = target_start_r + CPBox_row; reg_Rc = target_start_c + CPBox_col \n")
      f_all.write("# trg_Lr = abs(min(dR_row,0));         trg_Lc = abs(min(dR_col,0))         \n")
      f_all.write("# trg_Rr = source_start_r + CPBox_row; trg_Rc = source_start_c + CPBox_col \n")
      f_all.write("# Img[:,:] = 0.0;                                                          \n")
      f_all.write("# Img[trg_Lr:trg_Rr,trg_Lc:trg_Rc] = images[i][reg_Lr:reg_Rr,reg_Lc:reg_Rc]\n")
      f_all.write("# images[i][:,:] = Img[:,:]                                                \n")
      f_all.write("###########################################################################\n")
      f_all.write("*,"+str(gWcR)+","+str(gWcC)+","+str(Cen_row_min)+","+str(Cen_col_min)+","+str(Cen_row_max)+","+str(Cen_col_max)+"\n")

      for i in range(length):
         dR_row = Avg_Vect_all[i,0];    dR_col = Avg_Vect_all[i,1]
         CPBox_row = Row - abs(dR_row)        ### CPBox       : Corresponding Block Size
         CPBox_col = Col - abs(dR_col)        ###

         target_start_r = max(dR_row,0);    target_start_c = max(dR_col,0)
         target_end_r = target_start_r + CPBox_row ; target_end_c = target_start_c + CPBox_col

         source_start_r = abs(min(dR_row,0));  source_start_c = abs(min(dR_col,0))
         source_end_r = source_start_r + CPBox_row;   source_end_c = source_start_c + CPBox_col

         reg_Lr = target_start_r; reg_Lc = target_start_c; reg_Rr = target_end_r; reg_Rc = target_end_c
         trg_Lr = source_start_r; trg_Lc = source_start_c; trg_Rr = source_end_r; trg_Rc = source_end_c

         f_all.write(str(Avg_Vect_all[i,0])+","+str(Avg_Vect_all[i,1])+",")
         f_all.write(str(trg_Lr)+","+str(trg_Rr)+","+str(trg_Lc)+","+str(trg_Rc)+",")
         f_all.write(str(reg_Lr)+","+str(reg_Rr)+","+str(reg_Lc)+","+str(reg_Rc)+"\n")

if self_stop == 1:   ### self-detecting termination
   if tm_p > 0 :     ### if the termination before the preset iteration no
      ###tm_p += 1
      if show_sp > 0:
         print(f"tm_p = {tm_p} p = {p} preset max. iteration no. = {iteration_no}")
      preset_itno = str(iteration_no)+"LMC"
      tm_itno = str(tm_p)+"LMC"

      if mv_out_sp == 1:
         for i in range(tm_p):
            mv_filename = output_file + "." + str(i+1) + "Lmv.txt"
            new_mv_filename = mv_filename.replace(preset_itno,tm_itno)
            print(f"Original = {mv_filename} revised = {new_mv_filename}")
            os.rename(mv_filename, new_mv_filename)
         """
         new_mv_filename = mv_file_all.replace(preset_itno,tm_itno)
         print(f"Original = {mv_file_all} revised = {new_mv_filename}")
         os.rename(mv_file_all, new_mv_filename)
         """
      print(f"preset output_file = {output_file}  ",end="")
      new_filename = output_file.replace(preset_itno,tm_itno)
      print(f" revised file name = {new_filename}")
      os.rename(output_file, new_filename)

      print("")
      it_sp = tm_p

for i in range(it_sp):
   print(f"{i}: min_dist = {min_dist_list[i]} Peak's max = {P_max_list[i]}  anchor's peak_no =[",end="")
   for j in range(len(anchor_peak_no[i])):
      print(anchor_peak_no[i][j],end=", ")
   print("]")

for i in range(it_sp):
   result = 0
   zero_cp = 0
   for j in range(len(MV_List[i])):
      AA = abs(MV_List[i][j][0])+abs(MV_List[i][j][1])
      if AA == 0:
         zero_cp += 1
      result += AA
   moved_ratio = (length-zero_cp)/length
   if length-zero_cp > 0:
      avg_move = result/(length-zero_cp)
   else:
      avg_move = 0.0
   print(f"{i}: sum = {result} MV_frame_no = {length-zero_cp} ",end="")
   print(f"({moved_ratio:.2f}) Avg_move = {avg_move:.1f} ",end="")
   print("")

if w_max_no_sp == 1:
   w_max_file = mv_file +".w_max.txt"
   with open(w_max_file,'w') as f:
      for i in range(len(P_max_list)):
         ###print(f"{i}\t{P_max_list[i]}")
         print(f"{P_max_list[i]}")
         f.write(str(P_max_list[i])+"\n")

print("Compelte!")
if mv_show_sp == 1:
   print("Motion Vector")
   for i in range(len(MV_List[0])):
      print(f"{i}",end="\t")
      for j in range(tm_p):
         (Ix,Iy) = MV_List[j][i]
         print(f"[{Ix},{Iy}]",end=" ")
      print(f"=> [{Avg_Vect_all[i,0]},{Avg_Vect_all[i,1]}]")

   print("")

   plt.figure("Summarized Motion Vector",figsize=(12,6))
   plt.plot(Avg_Vect_all[:,0],Avg_Vect_all[:,1],'.k')
   plt.waitforbuttonpress()
   plt.close()

output_info_file = new_filename +".info"
import datetime

with open(output_info_file, "w") as f:
   x = str(datetime.datetime.now())+"\n"
   x += 'python3.12 '+' '.join(sys.argv) 
   x += '\n\n'
   f.write(x)
   with open(sys.argv[0],'r') as s_code:
      lines = s_code.readlines()
   for line in lines:
      f.write(line)

quit()

