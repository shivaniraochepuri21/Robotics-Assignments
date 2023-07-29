from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
import os
import scipy
import numpy as np
from scipy import signal
from numpy import linalg as LA
import matplotlib.pyplot as plt
import math

def opticalFlow_LKA(I1g, I2g, window_size, T=1e-2):
    #Roberts Derivative masks to interpret the optical flow equation
    #These masks are applied to im1 and im2 respectively
    #After the masks are applied, the responses are added to get f_x, f_y, f_t (first derivates)
    
    im1_derMat_x = np.array([[-1., 1.], [-1., 1.]])
    im1_derMat_y = np.array([[-1., -1.], [1., 1.]])
    im1_derMat_t = np.array([[-1., -1.], [-1., -1.]])#*.25
    im2_derMat_x = im1_derMat_x
    im2_derMat_y = im1_derMat_y
    im2_derMat_t = np.array([[1., 1.], [1., 1.]])#*.25
    
    # Implement Lucas Kanade for each point, calculate I_x, I_y, I_t
    m = 'same'

    #space convolution of image with the derivate matrices
    f_x = signal.convolve2d(I1g, im1_derMat_x, boundary='symm', mode=m)
    f_y = signal.convolve2d(I1g, im1_derMat_y, boundary='symm', mode=m)
    f_t = signal.convolve2d(I2g, im2_derMat_t, boundary='symm', mode=m) + signal.convolve2d(I1g, im1_derMat_t, boundary='symm', mode=m)
    
    u = np.zeros(I1g.shape)
    v = np.zeros(I1g.shape)
    
    #Within window, window_size * window_size
    #Calculate optical flow solution, that is [u,v] for each pixel
    #In Lucas-Kanade case calculate for each window, i.e, 3*3. Here, we took 15*15
    
    for i in range(w, I1g.shape[0]-w):
        for j in range(w, I1g.shape[1]-w):
            f_x1 = f_x[i-w:i+w+1, j-w:j+w+1]
            f_y1 = f_y[i-w:i+w+1, j-w:j+w+1]
            f_t1 = f_t[i-w:i+w+1, j-w:j+w+1]

            Ix = f_x1.flatten()
            Iy = f_y1.flatten()
            It = f_t1.flatten()
            
            # define A,b matrices in AU = b, where U = [u,v]
            i11 = Ix*It
            i12 = Iy*It
            I11 = Ix*Ix
            I12 = Ix*Iy
            I21 = Iy*Ix
            I22 = Iy*Iy
            b = np.array([-np.sum(i11),-np.sum(i12)])
            A = np.array([[np.sum(I11),np.sum(I12)],[np.sum(I21),np.sum(I22)]])
            a = A.T.dot(A)
            ue,D,ve = LA.svd(a)
            
            if np.min(D) > T:
                U = (LA.inv(A)).dot(b)
                u[i,j] = U[0]
                v[i,j] = U[1]
            else:
                u[i,j]=0
                v[i,j]=0                
    return (u,v)

def plots_optical_flow(im1,im2,u,v,area_default,arrow_thres,fig_size):
     
    figure = plt.figure(figsize = fig_size)
    ax = figure.add_subplot(3,2,1)
    ax.imshow(im1, cmap='gray')
    ax.set_title("Image 1")
    ax.axis('on')
    
    ax = figure.add_subplot(3,2,2)
    ax.imshow(im2, cmap='gray')
    ax.set_title("Image 2")
    ax.axis('on')
    
    ax = figure.add_subplot(3,2,3)
    ax.imshow(u, cmap='gray')
    ax.set_title("U")
    ax.axis('on')
    
    ax = figure.add_subplot(3,2,4)
    ax.imshow(v, cmap='gray')
    ax.set_title("V")
    ax.axis('on')
         
    ax = figure.add_subplot(3,2,5)
    ax.imshow(u*u + v*v, cmap='gray')
    ax.set_title("Magnitude = U^2 + V^2")
    ax.axis('on')
    
    ax = figure.add_subplot(3,2,6)
    ax.imshow(np.arctan2(v,u), cmap='gray')
    ax.set_title("arc(v/u)")
    ax.axis('on')

    figure = plt.figure(figsize = fig_size)
    ax = figure.add_subplot(1,2,1)
    ax.imshow(im1, cmap='gray')
    ax.set_title("Optical flow Arrows")
    
    key_points = cv2.goodFeaturesToTrack(im1, 200, 0.01, 10, 3)
    for i in key_points:
        x,y = i[0]
        y = int(y)
        x = int(x)
        ax.arrow(x,y,u[y,x],v[y,x], head_width = 2, head_length = 5, color = (1,0,0))
        
    ax = figure.add_subplot(1,2,2)
    ax.imshow( (u*u + v*v > arrow_thres), cmap='gray')
    ax.set_title("optical_flow mask")
    ax.axis('on')

    frame1 = imutils.resize(im1, width=500)
    frame2 = imutils.resize(im2, width=500)
	
    #gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray1 = frame1
    gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)
	
    #gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray2 = frame2
    gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)
	
	# compute the absolute difference between the current frame and first frame
    Delta = cv2.absdiff(gray1, gray2)
    Threshold = cv2.threshold(Delta, 25, 255, cv2.THRESH_BINARY)[1]
	# dilate the thresholded image to fill in holes, then find contours on thresholded image
    Threshold = cv2.dilate(Threshold, None, iterations = 5)
    C = cv2.findContours(Threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    C = imutils.grab_contours(C)
	
	# loop over the contours
    for c in C:
		# if the contour is too small, ignore it
        if cv2.contourArea(c) < area_default:
            continue
		# compute the bounding box for the contour, draw it on the frame,
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
	
    figure = plt.figure(figsize = fig_size)
    ax = figure.add_subplot(3,1,1)
    ax.imshow(frame2, cmap='gray')
    ax.set_title("frame")
    ax.axis('on')
    
    ax = figure.add_subplot(3,1,2)
    ax.imshow(Threshold, cmap='gray')
    ax.set_title("Threshold")
    ax.axis('on')
    
    ax = figure.add_subplot(3,1,3)
    ax.imshow(Delta, cmap='gray')
    ax.set_title("Delta")
    ax.axis('on')
    
    plt.show()
    return None

def segment(path, flag, w, arrow_thres, area_default):	    
	# if flag is 0, then we are reading from webcam
	if flag==0:
		vs = VideoStream(src=0).start()
		time.sleep(2.0)
		#vs = cv2.VideoCapture(0) 
	# otherwise, we are reading from a video file
	else:
		vs = cv2.VideoCapture('/home/shivani/CV/videoplayback.mp4')
	# initialize the first frame in the video stream
	firstFrame = None
	# loop over the frames of the video
	while (True):
		frame = vs.read()
		#print("this is frame",frame)
		if flag == 0:
			frame = frame
		else: 
			frame = frame[1]
		# if the frame could not be grabbed, then we have reached the end of the video
		if frame is None:
			print("110")
			break
		# resize the frame, convert it to grayscale, and blur it
		#frame = imutils.resize(f, width=500)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (21, 21), 0)
		# if the first frame is None, initialize it
		if firstFrame is None:
			firstFrame = gray
			continue
		# compute the absolute difference between the current frame and first frame
		Delta = cv2.absdiff(firstFrame, gray)
		Threshold = cv2.threshold(Delta, 25, 255, cv2.THRESH_BINARY)[1]
		# dilate the thresholded image to fill in holes, then find contours on thresholded image
		Threshold = cv2.dilate(Threshold, None, iterations=2)
		C = cv2.findContours(Threshold.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		C = imutils.grab_contours(C)

		Im1 = firstFrame / 255.
		Im2 = gray / 255.

		u,v = opticalFlow_LKA(Im1,Im2,w)
		
		# loop over the contours
		for c in C:
			# if the contour is too small, ignore it
			if cv2.contourArea(c) < area_default:
				continue
			# compute the bounding box for the contour, draw it on the frame,
			(x, y, w, h) = cv2.boundingRect(c)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		# show the frame and record if the user presses a key
		
		cv2.imshow("Frame", frame)
		cv2.imshow("Threshold", Threshold)
		cv2.imshow("Delta", Delta)
        
		figure = plt.figure(figsize = fig_size)
		ax = figure.add_subplot(1,2,1)
		ax.imshow(gray, cmap='gray')
		ax.set_title("Optical flow Arrows")
		
		kps = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10, 3)
		for i in kps:
			x,y = i[0]
			y = int(y)
			x = int(x)
			ax.arrow(x,y,u[y,x],v[y,x], head_width = 2, head_length = 5, color = (1,0,0))
        
		ax = figure.add_subplot(1,2,2)
		ax.imshow( (u*u + v*v > arrow_thres), cmap='gray')
		ax.set_title("optical_flow mask")
		ax.axis('on')
		plt.show()

		cv2.waitKey(1000)
		#cv2.imshow("Magnitude", (u*u + v*v > arrow_thres))
		key = cv2.waitKey(1) & 0xFF
		# if the `w` key is pressed, break from the loop and stop streaming
		if key == ord('w'):
			break

	# cleanup the camera and close any open windows
	if flag==0:
		vs.stop()
	else: 
		vs.release()
	cv2.destroyAllWindows()


Dir ='/home/shivani/CV/eval-data-gray/'
path = '/home/shivani/CV/videoplayback.mp4'

img_list = []
for folder in os.listdir(Dir):
    list_dir = os.listdir(os.path.join(Dir,folder)) 
    #read images in grayscale
    img1 = cv2.imread(os.path.join(Dir,folder,list_dir[0]),0)
    img2 = cv2.imread(os.path.join(Dir,folder,list_dir[1]),0)
    img_list.append([img1,img2])

# Calculate optical flow on images
cnt = 0
window_size = 15
w = int(window_size/2)
arrow_thres = 0.1
flag = 0 
fig_size = (16,16)
area_default = 500
#decomment this part to run for the image pairs provided in the dataset
#and uncomment the last line to not run webcam or videofile
"""
for ims in img_list:
	im1,im2 = ims
	# normalize pixels 
	Im1 = im1 / 255.
	Im2 = im2 / 255.
	u,v = opticalFlow_LKA(Im1,Im2,w)
	plots_optical_flow(im1,im2,u,v,area_default,arrow_thres,fig_size)
	print(im1.shape)
	print(im2.shape)
	cnt +=1
"""
#function that applies optical flow on a video
#flag is 0 for live capture from webcam
#flag is 1 for reading from a video file
segment(path, flag, w, arrow_thres, area_default)
