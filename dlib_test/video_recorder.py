from picamera import PiCamera
from time import sleep

camera = PiCamera()



camera.start_preview()
#camera.start_recording("/home/pi/Desktop/object_detection/dlib_test/video.h264")
#sleep(5)
#camera.stop_recording()
for i in range(5):
    sleep(1)
    camera.capture('/home/pi/Desktop/object_detection/dlib_test/video_frames/image%s.jpg' % i)
camera.stop_preview()