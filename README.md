# car-lab-pi-vision-system
Raspberry Pi vision system for object detection and tracking

Created: 7-16-2020



Current Version: 
Uses TensorFlow Lite to detect a single object then tracks the object using OpenCV tracker. Every n frames the detecter is used again to update tracker.


TODO:
- Track multiple objects
- Split object tracking between processing cores to increase speed
- Incorporate centroid tracker so that each object is giving unique ID

----------------------------------------------------------------------------

Current Hardware:
- Raspberry Pi 4
- Coral USB Accelerator (Edge TPU)
- Pi Camera

Current Software
- TensorFlow Lite (Object detection)
- OpenCV (Image manipulation / Object tracking)

Previous Software
- PIL (Image manipulation)
- dlib (Object tracking)

----------------------------------------------------------------------------


*Anvay don't roast me if I spell something wrong; there's no spell check.
