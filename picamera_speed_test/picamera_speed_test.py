import argparse
import io
import re
import time

import numpy as np
import picamera

from PIL import Image



CAMERA_WIDTH = int(640/2)
CAMERA_HEIGHT = int(480/2)

def main():
  # initialize variables to calculate FPS
  instantaneous_frame_rates = []

  with picamera.PiCamera(
      resolution=(CAMERA_WIDTH, CAMERA_HEIGHT), framerate=30) as camera:
    camera.start_preview() #alpha = 200
    try:
      stream = io.BytesIO()
      start_time = time.monotonic()
      
      for _ in camera.capture_continuous(
          stream, format='jpeg', use_video_port=True):
        stream.seek(0)

        frame_rate = 1/ ((time.monotonic() - start_time))

        print("FPS: " + str(frame_rate))
        
        stream.seek(0)
        stream.truncate()
        
        start_time = time.monotonic()
        

    finally:
      camera.stop_preview()


if __name__ == '__main__':
  main()
