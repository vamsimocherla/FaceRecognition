# USAGE
# python simple_request.py

# import the necessary packages
import requests
import json
import cv2

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://localhost:5000/recognize"
IMAGE_PATH = "static/img/frame.jpg"  # "jemma.png"

# load the input image and construct the payload for the request
#image = open(IMAGE_PATH, "rb").read()
#payload = {"image": image}
frame = cv2.imread(IMAGE_PATH)
(h, w) = frame.shape[:2]
ret, jpeg = cv2.imencode('.jpg', frame)

# submit the request
res = requests.post(KERAS_REST_API_URL, data=jpeg.tobytes())
detection = None
blob_size = (300, 300)
try:
    r = res.json()
    # ensure the request was sucessful
    if r["success"]:
        print(json.dumps(r["results"], sort_keys=True, indent=2, ensure_ascii=False))
        detections = r["results"]
        for detection in detections:
            left = int(detection['left'])
            top = int(detection['top'])
            right = int(detection['right'])
            bottom = int(detection['bottom'])

            '''
            small_frame = cv2.resize(frame, blob_size)
            cv2.rectangle(small_frame, (left, top), (right, bottom), (0, 255, 0), 4)
            cv2.imshow('small-image', small_frame)

            top = int(top * float(h / blob_size[0]))
            bottom = int(bottom * float(h / blob_size[0]))

            left = int(left * float(w / blob_size[1]))
            right = int(right * float(w / blob_size[1]))

            '''
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 4)
            cv2.imshow('image', cv2.resize(frame, (0, 0), fx=0.5, fy=0.5))
            cv2.waitKey(0)
        # loop over the predictions and display them
        # for (i, result) in enumerate(r["predictions"]):
        #    print("{}. {}: {:.4f}".format(i + 1, result["label"],
        #                                  result["probability"]))

    # otherwise, the request failed
    else:
        print("Request failed")
except:
    print(res)


def draw_frame(top, right, bottom, left, box_color, line_thickness, outer):
    x1, y1 = (left, top)
    x2, y2 = (left, bottom)
    x3, y3 = (right, top)
    x4, y4 = (right, bottom)

    h_line_length = int(abs((right - left) / 4))
    v_line_length = int(abs((bottom - top) / 4))

    cv2.line(frame, (x1, y1), (x1, y1 + v_line_length), box_color, line_thickness)  # -- top-left
    cv2.line(frame, (x1, y1), (x1 + h_line_length, y1), box_color, line_thickness)

    cv2.line(frame, (x2, y2), (x2, y2 - v_line_length), box_color, line_thickness)  # -- bottom-left
    cv2.line(frame, (x2, y2), (x2 + h_line_length, y2), box_color, line_thickness)

    cv2.line(frame, (x3, y3), (x3 - h_line_length, y3), box_color, line_thickness)  # -- top-right
    cv2.line(frame, (x3, y3), (x3, y3 + v_line_length), box_color, line_thickness)

    cv2.line(frame, (x4, y4), (x4, y4 - v_line_length), box_color, line_thickness)  # -- bottom-right
    cv2.line(frame, (x4, y4), (x4 - h_line_length, y4), box_color, line_thickness)

    return frame

