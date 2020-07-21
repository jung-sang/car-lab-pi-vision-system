import picamera

with picamera.PiCamera() as camera:
    camera.resolution = (640, 480)
    camera.start_recording('test_video_8.h264')
    camera.wait_recording(15)
    camera.stop_recording()