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


def annotate_objects(annotator, results, labels):
  """Draws the bounding box and label for each object in the results."""
  for obj in results:
    
    # Convert the bounding box figures from relative coordinates
    # to absolute coordinates based on the original resolution
    ymin, xmin, ymax, xmax = obj['bounding_box']
    xmin = int(xmin * CAMERA_WIDTH)
    xmax = int(xmax * CAMERA_WIDTH)
    ymin = int(ymin * CAMERA_HEIGHT)
    ymax = int(ymax * CAMERA_HEIGHT)

    # Overlay the box, label, and score on the camera preview
    annotator.bounding_box([xmin, ymin, xmax, ymax])
    annotator.text([xmin, ymin],
                   '%s\n%.2f' % (labels[obj['class_id']], obj['score']))
    
    # Overlay centroid position of each bounding box
    object_center_x = (xmin + xmax)/2
    object_center_y = (ymin + ymax)/2
    annotator.centroid(object_center_x, object_center_y)
    
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

  #t = None
  
  counter = 1

  start_time = time.monotonic()
  time_all = []
  
  t = None
  
  vs = VideoStream(usePiCamera=True).start()
  time.sleep(1.0)
  
  while True:
      
      total_time = (time.monotonic() - start_time)
      start_time = time.monotonic()
      print("FPS: " + str(1/(total_time)))#/counter)))
      #print(total_time)
      #time_all.append(total_time)
      #print(str(sum(time_all)/len(time_all)) + ", FPS: " + str(1/(sum(time_all)/len(time_all))))
      
      counter += 1
      
      frame = vs.read()
      frame = cv2.resize(frame, (input_width, input_height))
      (H, W) = frame.shape[:2]
      
      if t == None:
          
          #cv_rect = (100, 100, 100, 100)
          
          #image_detector = cv2.resize(frame,(input_width, input_height))
          image_detector = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          
          results = detect_objects(interpreter, image_detector, args.threshold)
          rects = get_rects(results)
          for i in np.arange(0,len(results)):
              #format bounding box coordinates
              box = np.array(rects[i])
              (startY, startX, endY, endX) = box.astype("int")
              cv_rect = (startX, startY, endX - startX, endY - startY)
              #print(startX, startY, endX, endY)
              
              #Note on tracker types:
              #KCF: Average speed, Average accuracy
              #MOSSE: High speed, low accuracy
              #MedianFlow: High speed, good accuracy only on slow moving objects (current best)
              
              t = cv2.TrackerMedianFlow_create()
              t.init(frame, cv_rect)
              
              cv2.rectangle(frame, (startX, startY), (endX , endY),(0, 255, 0), 2)
              
              # return active objects from the centroid tracker
              objects = ct.update(rects)
              
              for (objectID, centroid) in objects.items():
                  text = "ID {}".format(objectID)
                  #annotator.text([centroid[0],centroid[1]], text)
                  cv2.putText(frame, text, (centroid[0]-10, centroid[1]-10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0))
                  cv2.circle(frame, (centroid[0], centroid[1]), 5, (0, 255, 0))
          
          #cv2.rectangle(frame, (100, 100), (100, 100),(0, 255, 0), 2)
          
      else:
          (success, box) = t.update(frame)
            
          if success:
              cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box [1] + box[3])),(0, 255, 0), 2)
              
              objects = ct.update([[int(box[1]), int(box[0]), int(box[1] + box[3]), int(box [0] + box[2])]])
              
              for (objectID, centroid) in objects.items():
                  text = "ID {}".format(objectID)
                  #annotator.text([centroid[0],centroid[1]], text)
                  cv2.putText(frame, text, (centroid[0]-10, centroid[1]-10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0))
                  cv2.circle(frame, (centroid[0], centroid[1]), 5, (0, 255, 0))
              
          if (counter % 15) == 0:
              t = None
      
      frame = cv2.resize(frame, (640,480))
      cv2.imshow("Frame", frame)
      key = cv2.waitKey(1) & 0xFF
      
      
      
      if key == ord("q"):
          break
        
  cv2.destroyAllWindows()
  vs.stop
      
  
  
  

#   with picamera.PiCamera(
#       resolution=(CAMERA_WIDTH, CAMERA_HEIGHT), framerate=30) as camera:
#     #camera.start_preview() #alpha = 200
#     start_time = time.monotonic()
#     
#     
#     vs = VideoStream(src=0).start()
#     time.sleep(1.0)
#     
#     
#     try:
#       stream = io.BytesIO()
#       annotator = Annotator(camera)
#       
#       for _ in camera.capture_continuous(
#           stream, format='jpeg', use_video_port=True):
#           
#         test_time = (time.monotonic() - test_start_time)
#         test_time_all.append(test_time)
#         print(str(sum(test_time_all)/len(test_time_all)) + ", FPS: " + str(1/(sum(test_time_all)/len(test_time_all))))
#         
#         stream.seek(0)
#         
#         
# 
#         counter += 1
#         
#         frame = vs.read
# 
#                 
#         image = Image.open(stream).convert('RGB')
#         cv_img = np.asarray(image)
#         
#         annotator.clear()
#         
#         
# 
#         
# 
#         # if there are no trackes, first must try to detect objects
#         if t == None:
#             image = image.resize((input_width, input_height), Image.ANTIALIAS)
#             results = detect_objects(interpreter, image, args.threshold)
# 
#             rects = get_rects(results)
#             
#             for i in np.arange(0,len(results)):
#                 #format bounding box coordinates
#                 print("new tracker")
#                 box = np.array(rects[i])
#                 (startY, startX, endY, endX) = box.astype("int")
#                 cv_rect = (startX, startY, endX - startX, endY - startY)
#                 
#                 t = cv2.TrackerMOSSE_create()
#                 t.init(cv_img, cv_rect)
#                 
#                 annotator.bounding_box([startX, startY, endX, endY])
# 
#             #annotate_objects(annotator, results, labels)
#         
#                   
#         else:
#             
# 
#             (success, box) = t.update(cv_img)
#             
#             
#             
#             if success:
#                 #annotator.bounding_box([box[0], box[1], box[0] + box[2], box [1] + box[3]])
#                 cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box [1] + box[3])),(0, 255, 0), 2)
#                 
#             
#             
#             #if (counter % 40) == 0:
#                 #t = None
#   
#             
#         #elapsed_ms = (time.monotonic() - start_time) * 1000
#         #annotator.text([5, 0], '%.1f ms' % (elapsed_ms))
#         #frame_rate = 1/ ((time.monotonic() - start_time))
#         #start_time = time.monotonic()
#         #print(frame_rate)
#         
#         #calculate average FPS
#         #instantaneous_frame_rates.append(frame_rate)
#         #avg_frame_rate = sum(instantaneous_frame_rates)/len(instantaneous_frame_rates)
#         #print("FPS: " + str(avg_frame_rate))
#         #annotator.text([5, 15], '%.1f FPS' % (avg_frame_rate))
#         
#         
#         #annotator.clear()
#         annotator.update()
#         cv2.imshow("Frame", frame)
#         
#         stream.seek(0)
#         stream.truncate()
#         
#         test_start_time = time.monotonic()
# 
#     finally:
#       camera.stop_preview()
#       vs.stop()


if __name__ == '__main__':
  main()