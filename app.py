import streamlit as st
from PIL import Image
import cv2
import numpy as np



col1, col2 = st.beta_columns(2)


img1 = st.sidebar.file_uploader("background", type="jpg")#background
img2 = st.sidebar.file_uploader("mask", type="jpg")#mask


original = Image.open(img1)
original = original.resize((640, 480))
col1.header("background")
col1.image(original, use_column_width=True)

grayscale = Image.open(img2)
grayscale = grayscale.resize((640, 480))
col2.header("mask")
col2.image(grayscale, use_column_width=True)


#background
img1 = Image.open(img1)
img1 = np.array(img1.convert('RGB'))
img1 = cv2.cvtColor(img1,1)




#mask
img2 = Image.open(img2)
img2 = np.array(img2.convert('RGB'))
img2 = cv2.cvtColor(img2,1)




def remove_background(img):
    #== Parameters =======================================================================
    BLUR = 5
    CANNY_THRESH_1 = 10
    CANNY_THRESH_2 = 100
    MASK_DILATE_ITER = 20
    MASK_ERODE_ITER = 20
    MASK_COLOR = (0.0,0.0,0.0) # In BGR format
    
    #== Processing =======================================================================
    
    #-- Read image -----------------------------------------------------------------------
    img = img
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #-- Edge detection -------------------------------------------------------------------
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)
    
    #-- Find contours in edges, sort by area ---------------------------------------------
    contour_info = []
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    
    
    #-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
    # Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    for c in contour_info:
        cv2.fillConvexPoly(mask, c[0], (255))
    # cv2.fillConvexPoly(mask, max_contour[0], (255))
    
    #-- Smooth mask, then blur it --------------------------------------------------------
    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask
    
    #-- Blend masked img into MASK_COLOR background --------------------------------------
    mask_stack  = mask_stack.astype('float32') / 255.0          # Use float matrices, 
    img         = img.astype('float32') / 255.0                 #  for easy blending
    
    masked = (mask_stack * img) + ((1 - mask_stack) * MASK_COLOR) # Blend
    masked = (masked * 255).astype('uint8')                     # Convert back to 8-bit 
    return masked


img2=remove_background(img2)


height, width, channels = img2.shape
print(height, width, channels)


img1 = cv2.resize(img1, (640, 480)) 
img2 = cv2.resize(img2, (640//4, 480//4))

num_rows, num_cols = img2.shape[:2]

rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 45, 1)
img2 = cv2.warpAffine(img2, rotation_matrix, (num_cols, num_rows))




h, w ,channels= img2.shape
hh, ww ,channels= img1.shape

yoff = round((hh-h)/2)
xoff = round((ww-w)/2)
print(yoff,xoff)

# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = img2.shape


roi = img1[yoff:yoff+h, xoff:xoff+w]


# Now create a mask of logo and create its inverse mask also

img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 12, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# Now black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

# Put logo in ROI and modify the main image
dst = cv2.add(img1_bg,img2_fg)


# img1[0:rows, 0:cols ] = dst
img1[yoff:yoff+h, xoff:xoff+w] = dst


st.image(img1,caption="Result")
# cv2.imshow('res',img1)
# cv2.imwrite("result.jpg",img1 )
# cv2.waitKey(10000)
# cv2.destroyAllWindows()


