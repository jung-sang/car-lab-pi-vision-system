import argparse
import io
import re
import time

from annotation import Annotator

import numpy as np
import picamera

from PIL import Image

import dlib

CAMERA_WIDTH = int(640/2)
CAMERA_HEIGHT = int(480/2)

def main():
  # initialize variables to calculate FPS
  instantaneous_frame_rates = []
  
  counter = 0
  t = None
  #win = dlib.image_window()

  with picamera.PiCamera(
      resolution=(CAMERA_WIDTH, CAMERA_HEIGHT), framerate=25) as camera:
    camera.start_preview() #alpha = 200
    try:
      stream = io.BytesIO()
      annotator = Annotator(camera)
      
      for _ in camera.capture_continuous(
          stream, format='jpeg', use_video_port=True):
        stream.seek(0)
        
        start_time = time.monotonic() #start_time declaration moved to give a more accurate measurement to calculate FPS

        image = Image.open(stream).convert('RGB')
        dlib_img = np.asarray(image)
        
        annotator.clear()
        
        if t == None:
            t = dlib.correlation_tracker()
            dlib_rect = dlib.rectangle(0, 0, 100, 100)
            t.start_track(dlib_img, dlib_rect)
            
        else:
            t.update(dlib_img)
            pos = t.get_position()
            
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
            
            x = (startX + endX) / 2
            y = (startY + endY) / 2
            
            annotator.centroid(x, y)
            #annotator.clear()  
            #annotator.bounding_box([startX, startY, endX, endY])
  
            
        elapsed_ms = (time.monotonic() - start_time) * 1000
        annotator.text([5, 0], '%.1f ms' % (elapsed_ms))
        frame_rate = 1/ ((time.monotonic() - start_time))
        
        #calculate average FPS
        instantaneous_frame_rates.append(frame_rate)
        avg_frame_rate = sum(instantaneous_frame_rates)/len(instantaneous_frame_rates)
        print("FPS: " + str(avg_frame_rate))
        annotator.text([5, 15], '%.1f FPS' % (avg_frame_rate))
        
        #annotator.clear()
        annotator.update() 
        
        stream.seek(0)
        stream.truncate()
        
        

    finally:
      camera.stop_preview()


if __name__ == '__main__':
  main()
