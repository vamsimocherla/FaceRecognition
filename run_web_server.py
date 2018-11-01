# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import settings
import helpers
import flask
import redis
import uuid
import time
import json
import io
import cv2
import face_db
import imutils

# initialize our Flask application and Redis server
app = flask.Flask(__name__)
db = redis.StrictRedis(host=settings.REDIS_HOST,
                       port=settings.REDIS_PORT, db=settings.REDIS_DB)


def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image


def prepare_image_cv(image):
    # convert string data to numpy array
    image = np.fromstring(image, np.uint8)
    # convert numpy array to image
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # get image shape before resize
    (h, w) = image.shape[:2]
    # resize the image to blob size
    image = cv2.resize(image, face_db.blob_sizes[face_db.model_index])
    # use imutils to resize, maintaining aspect ratio
    # imutils.resize(image,
    #                height=face_db.blob_sizes[face_db.model_index][0],
    #                width=face_db.blob_sizes[face_db.model_index][1])
    # return the processed image
    return w, h, image


@app.route("/")
def homepage():
    return "Welcome to the PyImageSearch Keras REST API!"


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format and prepare it for
            # classification
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image = prepare_image(image,
                                  (settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT))

            # ensure our NumPy array is C-contiguous as well,
            # otherwise we won't be able to serialize it
            image = image.copy(order="C")

            # generate an ID for the classification then add the
            # classification ID + image to the queue
            k = str(uuid.uuid4())
            image = helpers.base64_encode_image(image)
            d = {"id": k, "image": image}
            db.rpush(settings.IMAGE_QUEUE, json.dumps(d))

            # keep looping until our model server returns the output
            # predictions
            while True:
                # attempt to grab the output predictions
                output = db.get(k)

                # check to see if our model has classified the input
                # image
                if output is not None:
                    # add the output predictions to our data
                    # dictionary so we can return it to the client
                    output = output.decode("utf-8")
                    data["predictions"] = json.loads(output)

                    # delete the result from the database and break
                    # from the polling loop
                    db.delete(k)
                    break

                # sleep for a small amount to give the model a chance
                # to classify the input image
                time.sleep(settings.CLIENT_SLEEP)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


@app.route("/recognize", methods=["POST"])
def recognize():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.data is not None:
            # read the input image
            image = flask.request.data
            # convert image to OpenCV format
            w, h, image = prepare_image_cv(image)
            # get the image shape
            image_shape = image.shape
            # generate an ID for the image
            image_id = str(uuid.uuid4())
            # serialize the image
            image = helpers.serialize(image)
            # push the data onto the queue
            image_metadata = {
                "id": image_id,
                "image": image,
                "shape": image_shape,
                "w": w,
                "h": h
            }
            db.rpush(settings.IMAGE_QUEUE, json.dumps(image_metadata))

            # keep looping until our model server returns the output
            # predictions
            while True:
                # attempt to grab the output predictions
                output = db.get(image_id)

                # check to see if our model has classified the input
                # image
                if output is not None:
                    # add the output predictions to our data
                    # dictionary so we can return it to the client
                    output = output.decode("utf-8")
                    data["results"] = json.loads(output)

                    # delete the result from the database and break
                    # from the polling loop
                    db.delete(image_id)
                    break

                # sleep for a small amount to give the model a chance
                # to classify the input image
                time.sleep(settings.CLIENT_SLEEP)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# for debugging purposes, it's helpful to start the Flask testing
# server (don't use this for production
if __name__ == "__main__":
    print("* Starting web service...")
    app.run()
