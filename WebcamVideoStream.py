# import the necessary packages
from threading import Thread
import cv2

class WebcamVideoStream:
    def __init__(self, src=0, img_res=(1280,720), name="WebcamVideoStream", playback=False):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        self.playback = playback
        ####################################
        ### Ubuntu Hack - to improve FPS ###
        ####################################
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, img_res[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, img_res[1])
        
        (self.grabbed, self.frame) = self.stream.read()

        if(self.playback == True):
            if(self.grabbed == False): # loop playback
                self.stream.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # initialize the thread name
        self.name = name

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True