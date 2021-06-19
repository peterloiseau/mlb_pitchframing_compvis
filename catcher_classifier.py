# -*- coding: utf-8 -*-
"""
Created on Tue May 25 17:10:45 2021

@author: Peter L'Oiseau
"""

import tensorflow.keras
import numpy as np
import cv2 as cv 
import os.path
import imutils
import pandas as pd
import argparse
from moviepy.editor import VideoFileClip
import time
from scipy.stats import beta

path = 'C:/Users/peter/Documents/baseball-databases/savant_video'
os.chdir(path)
pitch_info_df = pd.read_csv('full_pitch_info.csv',encoding='latin1')
pitch_info_df.head()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=50, help="minimum area size")
args = vars(ap.parse_args())
fin_preds = pd.DataFrame()

t0 = time.time()
for f in range(0,len(pitch_info_df['vid_name'])):
    t_int = time.time()
    vid = pitch_info_df['vid_name'][f]
    vs = cv.VideoCapture('vids/'+vid)
    clip = VideoFileClip('vids/'+vid)
    clip_dur = clip.duration
    num_frames = int(vs.get(cv.CAP_PROP_FRAME_COUNT))
    v_fps = num_frames/clip_dur
    
    # initialize the first frame in the video stream
    firstFrame = None
    TM_DATA = None
    model = None
    ret = None
    frame = None
    prediction = None
    key = None
     
    print('START VID%s'%f)
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)
    
    # Load the model
    model = tensorflow.keras.models.load_model('keras_model.h5', compile=False)
    
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    TM_DATA = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    l=0
    #reduce by 10 times the number of frames investigated in each video target frames in the middle of the video with a slight right skew
    tar_list = [round(x) for x in beta.rvs(7.5, 10, size = int((vs.get(cv.CAP_PROP_FRAME_COUNT))/10))*int(vs.get(cv.CAP_PROP_FRAME_COUNT))]
    while True:
      ret , frame = vs.read()
      frame = frame if args.get("video", None) is None else frame[1]
      # if the frame could not be grabbed, then we have reached the end of the video
      if frame is None:
          break
        # resize the frame, convert it to grayscale, and blur it
      frame = imutils.resize(frame, width=500)
      gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
      gray = cv.GaussianBlur(gray, (21, 21), 0)
      # if the first frame is None, initialize it
      if firstFrame is None:
          firstFrame = gray
          continue
      if l in tar_list:
          # compute the absolute difference between the current frame and
          # first frame
          frameDelta = cv.absdiff(firstFrame, gray)
          thresh = cv.threshold(frameDelta, 25, 255, cv.THRESH_BINARY)[1]
            # dilate the thresholded image to fill in holes, then find contours
            # on thresholded image
          thresh = cv.dilate(thresh, None, iterations=2)
          cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
                cv.CHAIN_APPROX_SIMPLE)
          cnts = imutils.grab_contours(cnts)
          #cnts_list = [cv2.boundingRect(c) for c in cnts]
          predictions = pd.DataFrame()
           # loop over the contours
          for c in cnts:
                # if the contour is too small, ignore it
              if cv.contourArea(c) < args["min_area"]:
                  continue
                # compute the bounding box for the contour
              (x, y, w, h) = cv.boundingRect(c)
              #make the contents of the contour an image, scale it
              #and send it to an array
              if x>80&x<380&y>60:
                  im = frame[y:y+h,x:x+w]
                  im = cv.resize(im, (224, 224))
                  image_array = np.asarray(im)
                  # Normalize the image
                  normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
                  # Load the image into the array
                  TM_DATA[0] = normalized_image_array
                  
                  prediction = model.predict(TM_DATA)
                  row = pd.DataFrame(np.append([prediction[0][prediction[0].argmax()],prediction[0].argmax()],[num_frames,v_fps,f,l,x,y,w,h])).transpose()
                  predictions = predictions.append(row)     
                  '''cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                  cv.putText(frame, "class: {}".format(prediction[0].argmax()), (x, y),
                        cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)'''
      
      #overlay the most likely object in each frame for helmet, mitt and other 
          for i in range(3):
              try:
                 pred_filt = predictions[predictions[1]==i] 
                 fin_row=pred_filt.iloc[[pred_filt[0].argmax()]]
                 fin_row.to_csv('track_data.csv', mode='a', index=False, header=False)
                 '''fin_preds=fin_preds.append(fin_row)
                 plot_x, plot_y, plot_w, plot_h = fin_row.loc[0,'2':'5']
                 cv.rectangle(frame, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), (0, 255, 0), 2)
                 cv.putText(frame, "class: {}".format(str(i)), (plot_x, plot_y),
                    cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                 print("class {}".format(i))
                 print(i, plot_x, plot_y, plot_w, plot_h)'''
              except:
                  pass
      l+=1
    print(time.time()-t_int)  
    
t1 = time.time()
total = t1-t0
print(total)
