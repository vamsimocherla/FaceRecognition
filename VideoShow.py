from threading import Thread
import cv2

class VideoShow:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, frame=None, scale_factor=1, shape=(1920, 1080, 3), display_name="VideoShow"):
        self.frame = frame
        self.stopped = False
        self.shape = shape
        self.scale_factor = scale_factor
        self.display_name = display_name

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
            if(self.frame is not None):
                if(self.frame.shape[0] == self.shape[0]):
                    self.frame = cv2.resize(self.frame, (0, 0),
                            fx=float(1.0/self.scale_factor), fy=float(1.0/self.scale_factor))
                    cv2.imshow(self.display_name, self.frame)
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        self.stopped = True
        print("[INFO] Waiting for thread " + self.current_thread() + " to stop...", end='\r')
        self.join()
        print("[INFO] Waiting for thread " + self.current_thread() + " to stop...done")
