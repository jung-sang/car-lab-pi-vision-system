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

CAMERA_WIDTH = int(640/2)
CAMERA_HEIGHT = int(480/2)

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

  t = None
  test_time_all = []
  counter = 0

  test_start_time = time.monotonic()
  

  with picamera.PiCamera(
      resolution=(CAMERA_WIDTH, CAMERA_HEIGHT), framerate=30) as camera:
    camera.start_preview() #alpha = 200
    start_time = time.monotonic()
    try:
      stream = io.BytesIO()
      annotator = Annotator(camera)
      
      for _ in camera.capture_continuous(
          stream, format='jpeg', use_video_port=True):
          
        test_time = (time.monotonic() - test_start_time)
        test_time_all.append(test_time)
        print(str(sum(test_time_all)/len(test_time_all)) + ", FPS: " + str(1/(sum(test_time_all)/len(test_time_all))))
        
        stream.seek(0)
        
        

        counter += 1
        

                
        image = Image.open(stream).convert('RGB')
        cv_img = np.asarray(image)
        
        annotator.clear()
        
        

        

        # if there are no trackes, first must try to detect objects
        if t == None:
            image = image.resize((input_width, input_height), Image.ANTIALIAS)
            results = detect_objects(interpreter, image, args.threshold)

            rects = get_rects(results)
            
            for i in np.arange(0,len(results)):
                #format bounding box coordinates
                print("new tracker")
                box = np.array(rects[i])
                (startY, startX, endY, endX) = box.astype("int")
                cv_rect = (startX, startY, endX - startX, endY - startY)
                
                t = cv2.TrackerMOSSE_create()
                t.init(cv_img, cv_rect)
                
                annotator.bounding_box([startX, startY, endX, endY])

            #annotate_objects(annotator, results, labels)
        
                  
        else:
            

            (success, box) = t.update(cv_img)
            
            
            
            if success:
                annotator.bounding_box([box[0], box[1], box[0] + box[2], box [1] + box[3]])
                #cv2.rectangle(cv_img, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box [1] + box[3])),(0, 255, 0), 2)
                
            
            
            #if (counter % 40) == 0:
                #t = None
  
            
        #elapsed_ms = (time.monotonic() - start_time) * 1000
        #annotator.text([5, 0], '%.1f ms' % (elapsed_ms))
        #frame_rate = 1/ ((time.monotonic() - start_time))
        #start_time = time.monotonic()
        #print(frame_rate)
        
        #calculate average FPS
        #instantaneous_frame_rates.append(frame_rate)
        #avg_frame_rate = sum(instantaneous_frame_rates)/len(instantaneous_frame_rates)
        #print("FPS: " + str(avg_frame_rate))
        #annotator.text([5, 15], '%.1f FPS' % (avg_frame_rate))
        
        
        #annotator.clear()
        annotator.update() 
        
        stream.seek(0)
        stream.truncate()
        
        test_start_time = time.monotonic()

    finally:
      camera.stop_preview()


if __name__ == '__main__':
  main()
