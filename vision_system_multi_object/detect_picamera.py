# python3
#
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example using TF Lite to detect objects with the Raspberry Pi camera."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tflite_runtime.interpreter import load_delegate #coral

from centroid_tracker import CentroidTracker


import argparse
import io
import re
import time

from annotation import Annotator

import numpy as np
import picamera

from PIL import Image
from tflite_runtime.interpreter import Interpreter

import dlib
import cv2
from imutils.video import VideoStream
import imutils

import multiprocessing

CAMERA_WIDTH = 300 #int(640/2)
CAMERA_HEIGHT = 300 #int(480/2)

ct = CentroidTracker()


def load_labels(path):
  """Loads the labels file. Supports files with or without index numbers."""
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labels = {}
    for row_number, content in enumerate(lines):
      pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labels[int(pair[0])] = pair[1].strip()
      else:
        labels[row_number] = pair[0].strip()
  return labels


def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()

  # Get all output details
  boxes = get_output_tensor(interpreter, 0)
  classes = get_output_tensor(interpreter, 1)
  scores = get_output_tensor(interpreter, 2)
  count = int(get_output_tensor(interpreter, 3))

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
          'bounding_box': boxes[i],
          'class_id': classes[i],
          'score': scores[i]
      }
      # simplifies program to only add frisbee objects to the results
      if classes[i]==33.0: # firsbee = 33.0
          results.append(result)
  return results

    
def get_rects(results):
    rects = []
    
    for obj in results:
        # Convert the bounding box figures from relative coordinates
        # to absolute coordinates based on the original resolution
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * CAMERA_WIDTH)
        xmax = int(xmax * CAMERA_WIDTH)
        ymin = int(ymin * CAMERA_HEIGHT)
        ymax = int(ymax * CAMERA_HEIGHT)
        
        rects.append([ymin, xmin, ymax, xmax])
        
    return rects

def start_tracker(cv_rect, frame, inputQueue, outputQueue):
    #Note on tracker types:
    #KCF: Average speed, Average accuracy
    #MOSSE: High speed, low accuracy
    #MedianFlow: High speed, good accuracy only on slow moving objects (current best)
  
    # initialize tracker
    t = cv2.TrackerKCF_create()
    t.init(frame, cv_rect)
    
    while True:
        frame = inputQueue.get()
        
        if frame is not None:
            (success, box) = t.update(frame)
            # if tracker was successful
            if success == False:
                print("IT WAS FALSE!!!!!!")
            if success:
                outputQueue.put((int(box[0]), int(box[1]), int(box[0] + box[2]), int(box [1] + box[3])))
            
              
def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model', help='File path of .tflite file.', required=True)
  parser.add_argument(
      '--labels', help='File path of labels file.', required=True)
  parser.add_argument(
      '--threshold',
      help='Score threshold for detected objects.',
      required=False,
      type=float,
      default=0.4)
  args = parser.parse_args()

  labels = load_labels(args.labels)
  interpreter = Interpreter(args.model, experimental_delegates=[load_delegate('libedgetpu.so.1.0')]) #coral
  interpreter.allocate_tensors()
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
  
  # initialize variables to calculate FPS
  instantaneous_frame_rates = []
  
  # initialize variable for tracker use
  counter = 1

  start_time = time.monotonic()
  
  t = None
  
  # begin video stream internally
  vs = VideoStream(usePiCamera=True).start()
  
  # uncomment next two lines for exporting video
  # fourcc = cv2.VideoWriter_fourcc(*'XVID')
  #out = cv2.VideoWriter('output2.avi', fourcc, 30.0, (640,480))
  
  # wait 1 second to give the camera time to adjust to lighting
  time.sleep(1.0)
  
  # initialize queues for object tracking
  inputQueues = []
  outputQueues = []
  
  # main loop
  while True:
      
      # calculating instantaneous FPS
      total_time = (time.monotonic() - start_time)
      start_time = time.monotonic()
      print("FPS: " + str(1/(total_time)))
      
      # Keep track of loop number
      counter += 1
      
      # get and resize current frame from camera
      frame = vs.read()
      frame = cv2.resize(frame, (input_width, input_height))
      (H, W) = frame.shape[:2]
      
      # if inputQueues is empty
      if len(inputQueues) == 0:
          
          # formating the frame as an RGB image for the TensorFlow detector
          image_detector = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          
          # get object detection results from TensorFlow Lite object detection model
          results = detect_objects(interpreter, image_detector, args.threshold)
          
          # get coordinates of bounding boxes
          rects = get_rects(results)
          
          # loops through results
          for i in np.arange(0,len(results)):
              #format bounding box coordinates for OpenCV tracker
              box = np.array(rects[i])
              (startY, startX, endY, endX) = box.astype("int")
              cv_rect = (startX, startY, endX - startX, endY - startY)
              
              # create new input and output queues
              iq = multiprocessing.Queue()
              oq = multiprocessing.Queue()
              inputQueues.append(iq)
              outputQueues.append(oq)
              
              # create daemon process for new tracker
              p = multiprocessing.Process(target = start_tracker, args = (cv_rect, frame, iq, oq))
              p.daemon = True
              p.start()
              # print(p)
              
              #Note on tracker types:
              #KCF: Average speed, Average accuracy
              #MOSSE: High speed, low accuracy
              #MedianFlow: High speed, good accuracy only on slow moving objects (current best)
              
              # initialize tracker
              # t = cv2.TrackerMedianFlow_create()
              # t.init(frame, cv_rect)
              
              # draw bounding box from the detector on frame
              cv2.rectangle(frame, (startX, startY), (endX , endY),(0, 255, 0), 2)
              
              # return active objects from the centroid tracker
              objects = ct.update(rects)
              
              # display object centroid on screen
              for (objectID, centroid) in objects.items():
                  text = "ID {}".format(objectID)
                  cv2.putText(frame, text, (centroid[0]-10, centroid[1]-10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0))
                  cv2.circle(frame, (centroid[0], centroid[1]), 5, (0, 255, 0))
                  
          
      # if a tracker has already been set up    
      else:
          
          for iq in inputQueues:
              iq.put(frame)
              
          for oq in outputQueues:
              (rect1, rect2, rect3, rect4) = oq.get()
              cv2.rectangle(frame, (rect1, rect2), (rect3, rect4), (0, 255, 0), 2)
              
#           # update the tracker is new frame and get new results
#           (success, box) = t.update(frame)
#           
#           # if tracker was successful
#           if success:
#               # draw bounding box; box format [xmin, ymin, width, height], cv2.rectangle format [xmin, ymin, xmax, ymax]
#               cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box [1] + box[3])),(0, 255, 0), 2)
#               
#               # update centroid tracker; centroid format [ymin, xmin, ymax, xmax]
#               # TODO: Fix formating!
#               objects = ct.update([[int(box[1]), int(box[0]), int(box[1] + box[3]), int(box [0] + box[2])]])
#               
#               # draw centorid
#               for (objectID, centroid) in objects.items():
#                   text = "ID {}".format(objectID)
#                   #annotator.text([centroid[0],centroid[1]], text)
#                   cv2.putText(frame, text, (centroid[0]-10, centroid[1]-10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0))
#                   cv2.circle(frame, (centroid[0], centroid[1]), 5, (0, 255, 0))
#           
          # Every n frames the tracker will be erased and the object detector will run again to re-initialize the tracker
          # n=15 for MedianFlow
          if (counter % 30) == 0:
              inputQueues = []
              outputQueues = []
              print(p)
              p.terminate()
              p.join()
      
      # resize frame for display
      frame = cv2.resize(frame, (640,480))
      
      # uncomment next time to export video
      # out.write(frame)
      
      cv2.imshow("Frame", frame)
      key = cv2.waitKey(1) & 0xFF
      
      
      # key "q" quits main loop
      if key == ord("q"):
          break
  
  # once out of main loop program ends
  cv2.destroyAllWindows()
  vs.stop


if __name__ == '__main__':
  main()
