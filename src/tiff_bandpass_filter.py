import cv2
import sys
import os.path, time
import os
from matplotlib import pyplot as plt
import numpy as np
import tifffile
from skimage import io

##########################################################
argc_no = len(sys.argv)
if (argc_no < 2):
   print("No input!");
   print(" -i input_file_name");
   print(" -start s_no (the start frame) ");
   print(" -end e_no (the start frame) ");
   print(" -cf cf_r (cutoff frequency (default 0.01))"); 
   print(" -show (no parmeter, show each frame after Butterworth filtering)");
   print(" -F_BP_out output background video")
   exit()

print("Butterworth Bandpass Filtering")
print("The program was updated in 2024.01.07.")


Colormap=['gray','Greys','ocean_r','winter','hot','bone','viridis','magma']
color_sp = 5 ## cmap=Colormap[color_sp]

################# Set parameters ################
input_file = " "
out_file_name = ""
bk_out_file_name =""

show_sp = 0
start_frame = 0  ;  end_frame = -1
L_st = 0         ;  L_end = 0
cf_r = 0.01

ngr_threshold = 0.001
######### Extract the input parameters from argv ###########
name_all_sp = 0
i=1

format_str = "float16"
level_up = 1    ## rise all negative values to non-negative
short_name = 1
F_BP_out_sp = 0

bk_out_sp = 0
smooth_sp = 1
start_no = 0

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
   elif (sys.argv[i] == "-format"):
      format_str = sys.argv[i+1]
      i += 2
   elif (sys.argv[i] == "-cf"):
      cf_r = (float)(sys.argv[i+1])
      i += 2
   elif (sys.argv[i] == "-show"):
      show_sp = 1
      i += 1
   elif (sys.argv[i] == "-bk_out"):
      bk_out_sp = 1
      i += 1
   elif (sys.argv[i] == "-F_BP_out"):
      F_BP_out_sp = 1
      i += 1   
   elif (sys.argv[i] == "-long_name"):
      short_name = 1
      i += 1
   elif (sys.argv[i] == "-float"):
      level_up = 0
      i += 1
   elif (sys.argv[i] == "-tiff_name"):
      out_file_name = sys.argv[i+1]
      i += 2
   elif (sys.argv[i] == "-bk_output"):
      bk_out_file_name = sys.argv[i+1]
      i += 2
   elif (sys.argv[i] == "-name_all"): 
      name_all_sp = 1
      i += 1 
   elif (sys.argv[i] == "-no_smooth"):
      smooth_sp = 0
      i += 1 
   elif (sys.argv[i] == "-nosmooth"):
      smooth_sp = 0
      i += 1 
   else:
      i += 1 
##################################################################################
###    Check input parameters ###########
##################################################################################

if (input_file == " "):
   print("No input")
   exit()

print("Cutoff frequency =", cf_r)

if level_up == 1:
   print("Output is to rise all negative values to all values as non-negative!")
else:
   print("The output file format is in a simple floating format (float16)")

try:
   InFile = tifffile.TiffFile(input_file)
except IOError:
   print(f"\nOpen file {input_file} error! \nQuit")
   quit()

length = len(InFile.pages)
if L_end == 0 or L_end > length:
   L_end = length

print("Input file = ",input_file)
f_size = os.path.getsize(input_file)
f_size = (f_size/1000000000)
print(f"File size = {f_size:.2f} GB" )

########## Check whether the input video with negative intensty values  ##################
print("Single Frame Read-Process-Write (RPW) Mode")

InFile = tifffile.TiffFile(input_file)
length = len(InFile.pages)
(Row, Col) = InFile.pages[0].shape
total_sum = Row*Col*length

input_type = (str)(InFile.pages[0].dtype)
print(f"data type = {input_type}")

star_no = 100
star_step = (int)((L_end-L_st)/star_no)
change_type = input_type

if "float" in input_type:
   print("Check the data range ")
   for i in range(star_no):
      print(".",end="",flush=True)
   for i in range(star_no):
      print("\x08",end="",flush=True)

   min_whole = np.min(InFile.pages[0].asarray())
   max_whole = np.max(InFile.pages[0].asarray())
   ng_sum = 0
   for i in range(L_st,L_end):
      if (i % star_step == 0):
         print(">",end="",flush=True)
      ng_sum += np.sum(InFile.pages[i].asarray() < 0)
      min_f = np.min(InFile.pages[i].asarray())
      max_f = np.max(InFile.pages[i].asarray())
      if min_whole > min_f:
         min_whole = min_f
      if max_whole > max_f:
         max_whole = max_f
   
   print("")
   ng_ratio = ng_sum/total_sum
   print(f"ng_ratio = {ng_ratio} max = {max_whole}  min = {min_whole}")

   dtype_uint16_max = 65535
   dtype_int16_max  = 32767; dtype_int16_min  = -32767

   print(f"max_whole - min_whole = {max_whole - min_whole} dtype_uint16_max = {dtype_uint16_max}")
   if (max_whole - min_whole) < dtype_uint16_max:
      if min_whole >= 0 and max_whole < dtype_uint16_max :
         change_type = "uint16"
      if min_whole < 0  and max_whole < dtype_int16_max :
         change_type = "int16"
   else:
      print(f"Out of int16 or uint16 ranges. Will skip the median smoothing.")
      change_type = "unknown"

fout_str = "f16"
if format_str == "float16":
   format = "e"
if format_str == "float32":
   format = "float32"
   fout_str = "f32"
if format_str == "float64":
   format = "float64"
   fout_str = "f64"
if format_str == "int32":
   format = "int32"
   fout_str = "i32"
if format_str == "int16":
   format = "int16"
   fout_str = "i16"

if level_up == 1:
   format = "uint16"
   fout_str = "u16_Zup"

out_sp = 0

print(f"smooth_sp = {smooth_sp}")

F_BP_file = ""
if out_file_name != "":
   bp_file = out_file_name
   out_sp = 1
else:
   if input_file[-5:] == ".tiff":
      input_file = input_file[:-5]
   if input_file[-4:] == ".tif":
      input_file = input_file[:-4]
   bp_file = input_file
  
   if short_name == 1:
      if change_type != "unknown" and smooth_sp == 1:
         ##F_BP_file = bp_file +".F_smBP";  bp_file+=".smBP"
         F_BP_file   = bp_file +".smBK"  ;  bp_file+=".smFG"
      else:
         ##F_BP_file = bp_file +".F_BP";  bp_file+=".BP"
         F_BP_file   = bp_file +".BK"  ;  bp_file+=".FG"
      F_BP_file += ".tiff"
      bp_file+=".tiff"
   else:
      if name_all_sp == 0:
         if change_type != "unknown" and smooth_sp == 1:
            bp_file+=".smFG"   ## bp_file+=".smBP"
         else:
            bp_file+=".FG"     ## bp_file+=".BP"
         bp_file+=".cf_"+str(cf_r)
      else:
         if (L_st == 0 and L_end == length):
            bp_file += ".all"
         else:
            bp_file += ".s"+str(L_st)+"-e"+str(L_end)
         bp_file+=".FG.cf_"+str(cf_r)  ### bp_file+=".BP.cf_"+str(cf_r)

      F_BP_file = bp_file +"."+fout_str+".BK.tiff"
      bp_file+="."+fout_str+".tiff"

print(f"Output file = {bp_file} F-BP file = {F_BP_file}")

if os.path.exists(bp_file):
   print("Output file exists! Remove and Create a new one!")
   os.remove(bp_file)

from skimage.filters import butterworth

print("Bandpass filtering by ",end="")
if change_type != "unknown" and smooth_sp == 1: 
   print("median filter and ",end="")
print(" Butterworth filter")

star_no = 100
star_step = (int)((L_end-L_st)/star_no)
print("")
for i in range(star_no):
   print(".",end="",flush=True)
for i in range(star_no):
   print("\x08",end="",flush=True)

frame = InFile.pages[0].asarray()
input_type = str(frame.dtype)
print(f"input_type = {input_type}")
  
bp_file_tmp = bp_file
if level_up == 1:
   Bd_min = 99999
   bp_file_tmp = bp_file_tmp[:-5] +".n_p.tiff"

print(f"L_st = {L_st} L_end = {L_end} smooth_sp = {smooth_sp}")
print(f"cf_r = {cf_r}")
with tifffile.TiffWriter(bp_file_tmp,bigtiff=True) as tif:
   for i in range(L_st,L_end):
      if star_step > 0:
         if (i % star_step == 0):
            print("@",end="",flush=True)
      frame = InFile.pages[i].asarray()
   
      if change_type != "unknown":
         frame = frame.astype(change_type)
      if smooth_sp == 1:
         frame = cv2.medianBlur(frame,5)

      bimage = butterworth(frame, cf_r, True, 8)
      
      if level_up == 1:
         bi_min = np.min(bimage)
         if bi_min < Bd_min:
            Bd_min = bi_min

      ##bimage = bimage.astype(format)
      bimage = bimage.astype(np.int16)
     
      tif.write(bimage,photometric='minisblack',contiguous=True)
   print("")

if F_BP_out_sp == 1:
   print(f"F-BP out file = {F_BP_file}")
   bpFile = tifffile.TiffFile(bp_file_tmp)
   with tifffile.TiffWriter(F_BP_file,bigtiff=True) as tif:
      for i in range(L_st,L_end):
         if star_step > 0:
            if (i % star_step == 0):
               print(">",end="",flush=True)
         org_frame = InFile.pages[i].asarray()
         bp_frame = bpFile.pages[i].asarray()
         f_bp_frame = org_frame-bp_frame
         ###f_bp_frame = org_frame.astype('int')-bp_frame.astype('int')
         f_bp_frame = f_bp_frame.astype('uint16')
         tif.write(f_bp_frame,photometric='minisblack',contiguous=True)
      print("\n")

if level_up == 1:
   ## Bd_min = (int)(Bd_min)
   print(f"Level up ... Bd_min = {Bd_min}")
   bpFile = tifffile.TiffFile(bp_file_tmp)
   with tifffile.TiffWriter(bp_file,bigtiff=True) as tif:
      for i in range(L_st,L_end):
         if star_step > 0:
            if (i % star_step == 0):
               print(".",end="",flush=True)
         frame = bpFile.pages[i].asarray() - Bd_min + 1
         frame = frame.astype('uint16')
         tif.write(frame,photometric='minisblack',contiguous=True)
      print("")
   #os.system("rm "+bp_file_tmp)
   os.remove(bp_file_tmp)

"""
if bk_out_sp == 1:
   print(f"Output background video {bk_out_file_name}")
   
   with tifffile.TiffWriter(bk_out_file_name,bigtiff=True) as tif:
"""
quit()

##################################################################################
### A. Bandpass filtering by two Butterworth filters
### https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.butterworth
### from skimage.filters import butterworth
###
###     |    ____________         |______________            |    __________
###     |   |                     |              |           |   |          |
###     |   |                     |              |           |   |          |
###     |   |                     |              |           |   |          |
###   --+---|------------->     --+---|----------|----->   --+---|----------|------>
###        cf_r                      cf_r     10*cf_r           cf_r     10*cf_r
###      High-pass                      Low-pass          Bandpass = High to Low pass
###
###I1 = butterworth(I0,cf_r,True,8)                               
###                                   I2 = butterworth(I1,10*cf_r,False,8)
###
###   fimages[i-L_st] = butterworth(images[i], cf_r, True, 8)
###   fimages[i-L_st] = butterworth(fimages[i-L_st], band_r*cf_r, False, 8)
###===============================================================================
### B. Bandpass filtering by median filter plus a Butterworth filter
###    I1 = cv2.medianBlur(I0,5)
###    I2 = butterworth(I0,cf_r,True,8)
### ps. 1. The noises can be removed but edges can be preserved by median filter
###     2. The decay rate for Butterworth filter is much  
##################################################################################
### skimage.filters.butterworth
### (image, cutoff_frequency_ratio=0.005, high_p
