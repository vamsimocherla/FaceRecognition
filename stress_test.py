# USAGE
# python stress_test.py

# import the necessary packages
from threading import Thread
import requests
import time
import cv2
import json

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://localhost:5000/recognize"
IMAGE_PATH = "static/img/frame.jpg"

# initialize the number of requests for the stress test along with
# the sleep amount between requests
NUM_REQUESTS = 50
SLEEP_COUNT = 0.01

def call_predict_endpoint(n):

    # ResNet
    # load the input image and construct the payload for the request
    # image = open(IMAGE_PATH, "rb").read()
    # payload = {"image": image}
    # submit the request
    # res = requests.post(KERAS_REST_API_URL, files=payload)

    # FaceRec
    frame = cv2.imread(IMAGE_PATH)
    ret, jpeg = cv2.imencode('.jpg', frame)
    # submit the request
    res = requests.post(KERAS_REST_API_URL, data=jpeg.tobytes())
    try:
        r = res.json()
        # ensure the request was sucessful
        if r["success"]:
            print("[INFO] thread {} OK".format(n))

        # otherwise, the request failed
        else:
            print("[INFO] thread {} FAILED".format(n))
    except:
        print("[INFO] thread {} FAILED with {}".format(n, str(res).strip("'<>() ").replace('\'', '\"')))


# loop over the number of threads
for i in range(0, NUM_REQUESTS):
    # start a new thread to call the API
    t = Thread(target=call_predict_endpoint, args=(i,))
    t.daemon = True
    t.start()
    time.sleep(SLEEP_COUNT)

# insert a long sleep so we can wait until the server is finished
# processing the images
time.sleep(5)
