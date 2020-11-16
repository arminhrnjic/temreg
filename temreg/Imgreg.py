"""
Image registration program for IL-AR-STEM purposes 

HOW TO USE:
- Run the program
- Select before image
- Select after image

TO DO:
- Plot images using matplotlib, not opencv.
- Make object-oriented software.
- Understand phase_cross_correlation algorithm and possibly replace it
- Sobel kernel for the detection of atom edges!
"""
# Imports:
import cv2
from tkinter import Tk
from tkinter import messagebox    
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
from skimage import registration
from skimage.transform import AffineTransform, warp

#######################################################################################

### Open dialog box for importing two images: Original and Shopped
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
messagebox.showinfo(title='Select original', message='Select the before image') # Messagebox for before image
filename_original = askopenfilename() # Importing original
messagebox.showinfo(title='Select shopped', message='Select the after image') # Messagebox for after image
filename_shopped = askopenfilename() # Importing Shopped

### Show original image
try: 
    image_original=cv2.cvtColor(cv2.imread(filename_original), cv2.COLOR_BGR2GRAY) #Import image and convert it to grayscale
except:
     messagebox.showinfo(title='No image selected', message='Images not selected properly') # Messagebox for error handle
# Testing part - commented
#cv2.imshow('Original', image_original)#Show image

### Show shopped image
image_shopped=cv2.cvtColor(cv2.imread(filename_shopped), cv2.COLOR_BGR2GRAY) #Import image and convert it to grayscale

# Testing part - commented
#cv2.imshow('Shopped', image_shopped) #Show image
#cv2.waitKey(600) # Hold open both images


######################################################################

""" 
Image registration algoritham. 
Relies heavily on phase_cross_correlation algorithm that I don't understand

"""

output=registration.phase_cross_correlation(image_original, image_shopped) #Image registration algorithm

print(output[0]) # Vector for image shifting (3D)
transform = AffineTransform(translation=[output[0][0],output[0][1]]) # Algorithm for shifting image (needs 2D vector)
shifted = warp(image_shopped, transform, mode='wrap', preserve_range=True) # No idea what this does
shifted = shifted.astype(image_shopped.dtype) # Createds an image file

# Testing part - commented
#cv2.imshow('Shifted', shifted) #Show image
#cv2.waitKey(600) # Holds the image on the screen for 600 microseconds 


######################################################################

# Subtraction and presenting images
sub = cv2.subtract(image_original, shifted) # Subtracts shifted from the original
cv2.imshow('Before', image_original) # Shows the original
cv2.imshow('After', shifted) # Shows shifted
cv2.imshow('Difference', sub) #Shows the final result of subtraction
cv2.waitKey(0) # Holds the image on the screen 