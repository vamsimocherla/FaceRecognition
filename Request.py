import datetime
import threading
import time
import requests
import cv2
import sys


class Request:

    def __init__(self, frame=None, interval=4):
        self.frame = frame
        self.interval = interval
        self.stopped = False
        self.faces = None

        ################
        ### Face API ###
        ################
        self.face_api_url = "http://localhost:5000/recognize"
        self.headers = {
            'Content-Type': 'application/octet-stream'
        }

    def start(self):
        threading.Thread(target=self.request_api, args=()).start()
        return self

    def current_milli_time(self):
        return int(round(time.time() * 1000))

    def request_api(self):
        data_time_millis = 0
        processing_time_millis = 0
        while not self.stopped:
            next_call = time.time()
            while True:
                if (self.frame is not None):
                    # compute data transfer time
                    start_time = self.current_milli_time()
                    # convert the image to jpeg format
                    ret, jpeg = cv2.imencode('.jpg', self.frame)
                    end_time = self.current_milli_time()
                    data_time_millis = "{:5.1f}".format(end_time - start_time)

                    # compute data response time
                    start_time = self.current_milli_time()
                    # send request and wait for response
                    response = None
                    try:
                        response = requests.post(self.face_api_url,
                                                 headers=self.headers,
                                                 data=jpeg.tobytes())
                    except:
                        print("POST error")
                        sys.exit(1)
                    end_time = self.current_milli_time()
                    processing_time_millis = "{:5.1f}".format(end_time - start_time)

                    try:
                        r = response.json()
                        # ensure the request was sucessful
                        if r["success"]:
                            self.faces = r["results"]
                        # otherwise, the request failed
                        else:
                            print("Request failed: {}".format(response))
                    except:
                        print("Request failed: {}".format(response))
                        sys.exit(1)

                else:
                    print("Frame is None {}".format(datetime.datetime.now()))

                print("[INFO] Transfer time: " + str(data_time_millis) + " ms"
                      + ", Processing time: " + str(processing_time_millis) + " ms", end='\r')

                next_call = next_call + self.interval;
                time.sleep(next_call - time.time())

    def stop(self):
        self.stopped = True
        print("[INFO] Waiting for thread " + self.current_thread() + " to stop...", end='\r')
        self.join()
        print("[INFO] Waiting for thread " + self.current_thread() + " to stop...done")
