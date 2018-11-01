# import the necessary packages
from keras.applications import ResNet50
from keras.applications import imagenet_utils
import numpy as np
import settings
import helpers
import redis
import time
import json
import face_recognition
import cv2
import face_db
import datetime

# connect to Redis server
db = redis.StrictRedis(host=settings.REDIS_HOST,
                       port=settings.REDIS_PORT, db=settings.REDIS_DB)


def classify_process():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    print("* Loading model...")
    model = ResNet50(weights="imagenet")
    print("* Model loaded")

    # continually pool for new images to classify
    while True:
        # attempt to grab a batch of images from the database, then
        # initialize the image IDs and batch of images themselves
        queue = db.lrange(settings.IMAGE_QUEUE, 0,
                          settings.BATCH_SIZE - 1)
        image_ids = []
        batch = None

        # loop over the queue
        for q in queue:
            # deserialize the object and obtain the input image
            q = json.loads(q.decode("utf-8"))
            image = helpers.base64_decode_image(q["image"],
                                                settings.IMAGE_DTYPE,
                                                (1, settings.IMAGE_HEIGHT,
                                                 settings.IMAGE_WIDTH,
                                                 settings.IMAGE_CHANS))

            # check to see if the batch list is None
            if batch is None:
                batch = image

            # otherwise, stack the data
            else:
                batch = np.vstack([batch, image])

            # update the list of image IDs
            image_ids.append(q["id"])

        # check to see if we need to process the batch
        if len(image_ids) > 0:
            # classify the batch
            print("* Batch size: {}".format(batch.shape))
            preds = model.predict(batch)
            results = imagenet_utils.decode_predictions(preds)

            # loop over the image IDs and their corresponding set of
            # results from our model
            for (imageID, resultSet) in zip(image_ids, results):
                # initialize the list of output predictions
                output = []

                # loop over the results and add them to the list of
                # output predictions
                for (imagenetID, label, prob) in resultSet:
                    r = {"label": label, "probability": float(prob)}
                    output.append(r)

                # store the output predictions in the database, using
                # the image ID as the key so we can fetch the results
                db.set(imageID, json.dumps(output))

            # remove the set of images from our queue
            db.ltrim(settings.IMAGE_QUEUE, len(image_ids), -1)

        # sleep for a small amount
        time.sleep(settings.SERVER_SLEEP)


class Server:

    # load our serialized model from disk
    print("[INFO] Loading CAFFE model...", end='\r')
    net = cv2.dnn.readNetFromCaffe(face_db.prototxt, face_db.model)
    print("[INFO] Loading CAFFE model...done")

    def recognize_face(self, crop_image):
        if (crop_image is None
                or crop_image.shape[0] == 0
                or crop_image.shape[1] == 0):
            return face_db.unknown_face
        # Convert the image from BGR color (which OpenCV uses)
        # to RGB color (which face_recognition uses)
        rgb_small_image = np.array(crop_image[:, :, ::-1])

        left = 0
        top = 0
        right = crop_image.shape[1]
        bottom = crop_image.shape[0]

        face_locations = [(top, right, bottom, left)]  # top right bottom left
        face_encodings = face_recognition.face_encodings(rgb_small_image, face_locations)
        face_encoding = face_encodings[0]
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(face_db.known_face_encodings, face_encoding)
        distances = face_recognition.face_distance(face_db.known_face_encodings, face_encoding)
        face = face_db.unknown_face

        # If a match was found in known_face_encodings, use the nearest one
        if True in matches:
            min_distance_index = np.argmin(distances)
            face = face_db.known_face_names[min_distance_index]

        return face

    def process_image(self, image, w, h, recognize=True):
        # Resize image of video to 1/scale_factor size for faster face recognition processing
        image = cv2.resize(image, (0, 0),
                           fx=float(1.0 / face_db.scale_factor),
                           fy=float(1.0 / face_db.scale_factor))

        (height, width) = image.shape[:2]
        # blob_image = cv2.resize(image, face_db.blob_sizes[face_db.model_index])
        blob_image = image
        blob = cv2.dnn.blobFromImage(blob_image, 1.0, face_db.blob_sizes[face_db.model_index],
                                     face_db.mean_val[face_db.model_index])

        # pass the blob through the network and obtain the
        # detections and predictions
        start_time = self.current_milli_time()
        self.net.setInput(blob)
        detections = self.net.forward()
        end_time = self.current_milli_time()
        detect_time = "{:5.1f}".format(end_time - start_time)
        print(" DetectTime: {}".format(detect_time), end='', flush=True)

        scale_y = float(h / face_db.blob_sizes[face_db.model_index][1])
        scale_x = float(w / face_db.blob_sizes[face_db.model_index][0])

        faces = []
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence < face_db.confidence:
                continue

            # compute the (x, y)-coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (left, top, right, bottom) = box.astype("int")

            # crop the face from the original image
            crop_image = image[top:bottom, left:right]

            face = face_db.unknown_face
            if recognize == True:
                # perform face recognition on the cropped image
                start_time = self.current_milli_time()
                face = self.recognize_face(crop_image)
                end_time = self.current_milli_time()
                rec_time = "{:5.1f}".format(end_time - start_time)
                print(" RecTime: {}".format(rec_time), end='\r', flush=True)
            else:
                print(" RecTime: {}".format(0), end='\r', flush=True)

            # gather the metadata
            metadata = {}
            metadata['face'] = face
            metadata['confidence'] = str(confidence)

            # scale the points back to original image size
            top = int(top * scale_y)
            bottom = int(bottom * scale_y)
            left = int(left * scale_x)
            right = int(right * scale_x)

            metadata['left'] = str(left)
            metadata['top'] = str(top)
            metadata['right'] = str(right)
            metadata['bottom'] = str(bottom)

            faces.append(metadata)

        if len(faces) == 0:
            print(" RecTime: {}".format(0), end='\r', flush=True)

        return faces

    def recognize(self):
        # continually pool for new images
        while True:
            # attempt to grab a batch of images from the database, then
            # initialize the image IDs and batch of images themselves
            queue = db.lrange(settings.IMAGE_QUEUE, 0,
                              settings.BATCH_SIZE - 1)
            # loop over the queue
            for q in queue:
                start_time = self.current_milli_time()
                # deserialize the payload object and obtain the input image
                image_metadata = json.loads(q.decode("utf-8"))
                end_time = self.current_milli_time()
                ser_time = "{:5.1f}".format(end_time - start_time)

                # deserialize the input image
                start_time = self.current_milli_time()
                image_id = image_metadata["id"]
                img_bytes = image_metadata["image"]
                img_shape = image_metadata["shape"]
                w = image_metadata["w"]
                h = image_metadata["h"]
                image = helpers.deserialize(img_bytes, img_shape)
                end_time = self.current_milli_time()
                deser_time = "{:5.1f}".format(end_time - start_time)

                # print("* ID: {}, Image size: {}".format(image_id, image.shape))
                print("* SerTime: {}, DeSerTime: {}".format(ser_time, deser_time), end='', flush=True)

                # process input image for faces
                output = self.process_image(image, w, h, recognize=True)

                # store the output predictions in the database, using
                # the image ID as the key so we can fetch the results
                db.set(image_id, json.dumps(output))

                # remove the set of images from our queue
                db.ltrim(settings.IMAGE_QUEUE, 1, -1)

            # sleep for a small amount
            time.sleep(settings.SERVER_SLEEP)

    def recognize_batch_cpu(self, recognize=True):
        # continually pool for new images
        while True:
            # attempt to grab a batch of images from the database, then
            # initialize the image IDs and batch of images themselves
            queue = db.lrange(settings.IMAGE_QUEUE, 0,
                              settings.BATCH_SIZE - 1)
            image_ids = []
            images = []
            ws = []
            hs = []
            faces = []

            # loop over the queue
            for q in queue:
                # deserialize the object and obtain the input image
                # deserialize the payload object and obtain the input image
                image_metadata = json.loads(q.decode("utf-8"))

                # deserialize the input image
                image_id = image_metadata["id"]
                img_bytes = image_metadata["image"]
                img_shape = image_metadata["shape"]
                w = image_metadata["w"]
                h = image_metadata["h"]
                image = helpers.deserialize(img_bytes, img_shape)
                # update the batch of images
                images.append(image)
                # update the list of image IDs
                image_ids.append(image_id)
                ws.append(w)
                hs.append(h)
                faces.append([])

            # check to see if we need to process the batch
            if len(image_ids) > 0:
                # classify the batch
                print("[INFO] [{}]  Batch size: {} x {} ".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                        len(images),
                                                        images[0].shape), end='', flush=True)
                # convert the images to a OpenCV compatible blob
                # SSD output is 1 - by - 1 - by - ndetections - by - 7
                blob = cv2.dnn.blobFromImages(images,
                                              scalefactor=1.0,
                                              size=face_db.blob_sizes[face_db.model_index],
                                              mean=face_db.mean_val[face_db.model_index])
                # pass the blob through the network and obtain the
                # detections and predictions
                start_time = self.current_milli_time()
                self.net.setInput(blob)
                detections = self.net.forward()
                end_time = self.current_milli_time()
                detect_time = "{:5.1f}".format(end_time - start_time)
                if(recognize == True):
                    print(" Detection Time: {} ".format(detect_time), end='', flush=True)
                else:
                    print(" Detection Time: {} ".format(detect_time))

                start_time = self.current_milli_time()

                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                # detection = [img_id, class_id, confidence, left, bottom, right, top]
                detections = detections[:, :, detections[0, 0, :, 2] > face_db.confidence, :]

                # TODO: run the Face Recognition part in batches
                for i in range(0, detections.shape[2]):
                    # extract the image_id in the batch
                    image_id = int(detections[0, 0, i, 0])
                    # extract the confidence associated with the prediction
                    confidence = detections[0, 0, i, 2]
                    # get the image shape
                    (h, w) = images[image_id].shape[:2]
                    # compute the (x, y)-coordinates of the bounding box
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (left, top, right, bottom) = box.astype("int")
                    # assign face to unknown
                    face = face_db.unknown_face
                    # crop the face from the original image
                    crop_image = images[image_id][top:bottom, left:right]
                    if recognize == True:
                        # perform face recognition on the cropped image
                        face = self.recognize_face(crop_image)
                    # gather the metadata
                    metadata = {}
                    metadata['face'] = face
                    metadata['confidence'] = str(confidence)

                    # compute resize scale on both X and Y axes
                    scale_y = float(hs[image_id] / face_db.blob_sizes[face_db.model_index][1])
                    scale_x = float(ws[image_id] / face_db.blob_sizes[face_db.model_index][0])

                    # scale the points back to original image size
                    top = int(top * scale_y)
                    bottom = int(bottom * scale_y)
                    left = int(left * scale_x)
                    right = int(right * scale_x)

                    metadata['left'] = str(left)
                    metadata['top'] = str(top)
                    metadata['right'] = str(right)
                    metadata['bottom'] = str(bottom)

                    faces[image_id].append(metadata)

                if recognize == True:
                    end_time = self.current_milli_time()
                    rec_time = "{:5.1f}".format(end_time - start_time)
                    print(" RecTime: {}".format(rec_time))

                for i in range(len(image_ids)):
                    # store the output predictions in the database, using
                    # the image ID as the key so we can fetch the results
                    db.set(image_ids[i], json.dumps(faces[i]))

                # remove the set of images from our queue
                db.ltrim(settings.IMAGE_QUEUE, len(image_ids), -1)

            # sleep for a small amount
            time.sleep(settings.BATCH_SERVER_SLEEP)

    def recognize_batch_gpu(self):
        # continually pool for new images
        while True:
            # attempt to grab a batch of images from the database, then
            # initialize the image IDs and batch of images themselves
            queue = db.lrange(settings.IMAGE_QUEUE, 0,
                              settings.BATCH_SIZE - 1)
            image_ids = []
            images = []

            ws = []
            hs = []

            # loop over the queue
            for q in queue:
                # deserialize the object and obtain the input image
                # deserialize the payload object and obtain the input image
                image_metadata = json.loads(q.decode("utf-8"))

                # deserialize the input image
                image_id = image_metadata["id"]
                img_bytes = image_metadata["image"]
                img_shape = image_metadata["shape"]
                w = image_metadata["w"]
                h = image_metadata["h"]
                image = helpers.deserialize(img_bytes, img_shape)
                # update the batch of images
                images.append(image)
                # update the list of image IDs
                image_ids.append(image_id)
                ws.append(w)
                hs.append(h)

            # check to see if we need to process the batch
            if len(image_ids) > 0:
                start_time = self.current_milli_time()
                # classify the batch
                print("* Batch size: {} x {} ".format(len(images), images[0].shape), end='', flush=True)
                results = face_recognition.batch_face_locations(images, number_of_times_to_upsample=0)
                end_time = self.current_milli_time()
                detect_time = "{:5.1f}".format(end_time - start_time)
                print(" Detection Time: {}".format(detect_time))


                # loop over the image IDs and their corresponding set of
                # results from our model
                for (frame_id, face_locations) in enumerate(results):
                    image_id = image_ids[frame_id]
                    scale_y = float(hs[frame_id] / face_db.blob_sizes[face_db.model_index][1])
                    scale_x = float(ws[frame_id] / face_db.blob_sizes[face_db.model_index][0])

                    # initialize the list of output predictions
                    output = []

                    # loop over the results and add them to the list of
                    # output predictions
                    for face_location in face_locations:
                        top, right, bottom, left = face_location
                        # gather the metadata
                        metadata = {}
                        metadata['face'] = face_db.unknown_face
                        metadata['confidence'] = str(0)

                        # scale the points back to original image size
                        top = int(top * scale_y)
                        bottom = int(bottom * scale_y)
                        left = int(left * scale_x)
                        right = int(right * scale_x)

                        metadata['left'] = str(left)
                        metadata['top'] = str(top)
                        metadata['right'] = str(right)
                        metadata['bottom'] = str(bottom)
                        output.append(metadata)

                    # store the output predictions in the database, using
                    # the image ID as the key so we can fetch the results
                    db.set(image_id, json.dumps(output))

                # remove the set of images from our queue
                db.ltrim(settings.IMAGE_QUEUE, len(image_ids), -1)

            # sleep for a small amount
            time.sleep(settings.BATCH_SERVER_SLEEP)

    def current_milli_time(self):
        return int(round(time.time() * 1000))


# if this is the main thread of execution start the model server
# process
if __name__ == "__main__":
    server = Server()
    # server.recognize()
    server.recognize_batch_cpu(recognize=False)
    # server.recognize_batch_gpu()
#    classify_process()
