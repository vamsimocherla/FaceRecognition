import face_recognition
import sys
import zmq
import multiprocessing as mp
import threading
import time
import numpy as np
import base64
from SerDe import SerDe
import cv2
import json

def tprint(msg):
    """like print, but won't get newlines confused with multiple threads"""
    sys.stdout.write(msg + '\n')
    sys.stdout.flush()

class ServerTask(threading.Thread):
#class ServerTask(mp.Process):
    """ServerTask"""
    def __init__(self, num_workers=3):
        self.num_workers = num_workers
        threading.Thread.__init__ (self)
        #mp.Process.__init__ (self)

    def run(self):
        context = zmq.Context()
        frontend = context.socket(zmq.ROUTER)
        frontend.bind("tcp://127.0.0.1:5559")

        backend = context.socket(zmq.DEALER)
        backend.bind("tcp://127.0.0.1:5560")

        workers = []
        for i in range(self.num_workers):
            worker = ServerWorker(context, i)
            worker.start()
            workers.append(worker)

        zmq.proxy(frontend, backend)

        frontend.close()
        backend.close()
        context.term()

class ServerWorker(threading.Thread):
#class ServerWorker(mp.Process):
    
    # Serializer-Deserializer
    serde = SerDe()
    
    # Image DB location
    img_dir = "static/img/"
    model_dir = "static/models/"

    # Load a third sample picture and learn how to recognize it.
    john_linss_image = face_recognition.load_image_file(img_dir + "john_linss.jpg")
    john_linss_face_encoding = face_recognition.face_encodings(john_linss_image)[0]

    # Load a fifth sample picture and learn how to recognize it.
    jones_image = face_recognition.load_image_file(img_dir + "jan_jones.png")
    jones_face_encoding = face_recognition.face_encodings(jones_image)[0]

    # Load a seventh sample picture and learn how to recognize it.
    williamshen_image = face_recognition.load_image_file(img_dir + "William_Shen.jpg")
    williamshen_face_encoding = face_recognition.face_encodings(williamshen_image)[0]

    # Load a eigth sample picture and learn how to recognize it.
    yusukewatanabe_image = face_recognition.load_image_file(img_dir + "Yusuke_Watanabe.jpg")
    yusukewatanabe_face_encoding = face_recognition.face_encodings(yusukewatanabe_image)[0]

    # Create arrays of known face encodings and their names
    known_face_encodings = [
        john_linss_face_encoding,
        jones_face_encoding,
        williamshen_face_encoding,
        yusukewatanabe_face_encoding
    ]
    known_face_names = {
        0: { "status": "See Ambassador", "jStatus": "アンバサダーを参照", "img": img_dir + "john_linss.jpg", "name": "John Linss", "username": "jlinss", "age": "21", "gender": "Male", "tier": "Platinum Member", "jTier": "プラチナ会員", "visits": "1,3", "residency": "Resident", "jResidency": "居住者", "prefix": "Mr."},
        1: { "status": "See Ambassador", "jStatus": "アンバサダーを参照", "img": img_dir + "jan_jones.png", "name": "Jan Jones Blackhurst", "username": "jblackhurst", "age": "21", "gender": "Female", "tier": "Seven Star Member", "jTier": "セブンスター会員", "visits": "1,2", "residency": "Non-resident", "jResidency": "非居住者", "prefix": "Mrs."},
        2: { "status": "See Ambassador", "jStatus": "アンバサダーを参照", "img": img_dir + "William_Shen.jpg", "name": "William Shen", "username": "wshen", "age": "28", "gender": "Male", "tier": "Platinum Member", "jTier": "プラチナ会員", "visits": "0,2", "residency": "Resident", "jResidency": "居住者", "prefix": "Mr."},
        3: { "status": "See Ambassador", "jStatus": "アンバサダーを参照", "img": img_dir + "Yusuke_Watanabe.jpg", "name": "Yusuke Watanabe", "username": "ywatanabe", "age": "26", "gender": "Male", "tier": "Seven Star Member", "jTier": "セブンスター会員", "visits": "1,3", "residency": "Resident", "jResidency": "居住者", "prefix": "Mr."}
    }
    unknown_face = { "name": "Unknown" }

    # Input Video Props
    scale_factor = 1

    # TODO: try different pre-trained models
    models = [
    "res10_300x300_ssd_iter_140000/"
    ]
    blob_sizes = [
    (500, 500)
    ]
    mean_val = [
    (104.0, 177.0, 123.0)
    ]
    model_index = 0 % len(models)
    model_version = models[model_index]
    
    prototxt = model_dir + model_version + "deploy.prototxt.txt"
    model = model_dir + model_version + "model.caffemodel"
    confidence = 0.50

    """ServerWorker"""
    def __init__(self, context, id):
        self.id = id
        
        threading.Thread.__init__ (self)
        #mp.Process.__init__ (self)
        
        self.context = context
        # load our serialized model from disk
        print("[INFO] Worker-%s => Loading CAFFE model..." % (self.id), end='\r')
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt, self.model)
        print("[INFO] Worker-%s => Loading CAFFE model...done" % (self.id))

    def recognize_face(self, crop_frame):

        if(crop_frame is None
            or crop_frame.shape[0]==0
            or crop_frame.shape[1]==0):
            return self.unknown_face
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = crop_frame[:, :, ::-1]

        startX = 0
        startY = 0
        endX = crop_frame.shape[1]
        endY = crop_frame.shape[0]
        
        left = startX
        top = startY
        right = endX
        bottom = endY

        face_locations = [(top, right, bottom, left)] # top right bottom left
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_encoding = face_encodings[0]
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
        distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
        face = self.unknown_face
        distance = 1

        # If a match was found in known_face_encodings, use the nearest one
        if True in matches:
            min_distance_index = np.argmin(distances)
            face = self.known_face_names[min_distance_index]
            distance = distances[min_distance_index]

        return face

    def process_frame_fast(self, frame, recognize=True):
        # Resize frame of video to 1/scale_factor size for faster face recognition processing
        frame = cv2.resize(frame, (0, 0), fx=float(1.0/self.scale_factor), fy=float(1.0/self.scale_factor))

        (h, w) = frame.shape[:2]
        blob_frame = cv2.resize(frame, self.blob_sizes[self.model_index])
        blob = cv2.dnn.blobFromImage(blob_frame, 1.0, self.blob_sizes[self.model_index], self.mean_val[self.model_index])
     
        # pass the blob through the network and obtain the
        # detections and predictions
        self.net.setInput(blob)
        detections = self.net.forward()
        faces = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence < self.confidence:
                continue

            # compute the (x, y)-coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (left, top, right, bottom) = box.astype("int")
            
            # crop the face from the original image
            crop_frame = frame[top:bottom, left:right]
            bounding_box = (left, top, right, bottom)

            face = self.unknown_face
            if(recognize == True):
                # perform face recognition on the cropped image
                face = self.recognize_face(crop_frame)
            
            # gather the metadata
            metadata = {}
            metadata['face'] = face
            metadata['confidence'] = str(confidence)
            metadata['left'] = str(left)
            metadata['top'] = str(top)
            metadata['right'] = str(right)
            metadata['bottom'] = str(bottom)
            
            faces.append(metadata)
            
        return faces

    def process_frame(self, frame, model="hog", recognize=True):
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=float(1.0/self.scale_factor), fy=float(1.0/self.scale_factor))

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        self.face_locations = face_recognition.face_locations(rgb_small_frame, model=model)
        self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
        faces = []
        for i in range(len(self.face_encodings)):
        
            (top, right, bottom, left) = self.face_locations[i]
            face_encoding = self.face_encodings[i]
            face = self.unknown_face
            confidence = 1
            
            if(recognize == True):
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                distance = 1

                # If a match was found in known_face_encodings, use the nearest one
                if True in matches:
                    min_distance_index = np.argmin(distances)
                    face = self.known_face_names[min_distance_index]
                    distance = distances[min_distance_index]
                    confidence = float(1.0/(distance+1.0))

            # gather the metadata
            metadata = {}
            metadata['face'] = face
            metadata['confidence'] = str(confidence)
            metadata['left'] = str(left)
            metadata['top'] = str(top)
            metadata['right'] = str(right)
            metadata['bottom'] = str(bottom)
            
            faces.append(metadata)
        return faces

    def run(self):
        worker = self.context.socket(zmq.DEALER)
        worker.connect("tcp://127.0.0.1:5560")
        tprint('[INFO] Worker-%s started' % (self.id))
        while True:
            ident, frame_metadata_bytes, frame_bytes = worker.recv_multipart()
            frame_metadata = json.loads(frame_metadata_bytes.decode('utf8'))
            frame = self.serde.deserialize(frame_bytes,
                dtype=frame_metadata['dtype'], shape=frame_metadata['shape'])
            
            json_response = []
            # process the frame
            if(frame_metadata['fast'] == False):
                json_response = self.process_frame(frame,
                    model=frame_metadata['model'],
                    recognize=frame_metadata['face_rec'])
            else:
                json_response = self.process_frame_fast(frame,
                    recognize=frame_metadata['face_rec'])
            
            # send the list of bounding boxes
            # and corresponding faces back
            worker.send_multipart([ident, bytes(json.dumps(json_response), 'UTF-8')])

        worker.close()

class FrameProcessorServer(mp.Process):

    serde = SerDe()
    
    # Image DB location
    img_dir = "static/img/"
    model_dir = "static/models/"

    # Load a third sample picture and learn how to recognize it.
    john_linss_image = face_recognition.load_image_file(img_dir + "john_linss.jpg")
    john_linss_face_encoding = face_recognition.face_encodings(john_linss_image)[0]

    # Load a fifth sample picture and learn how to recognize it.
    jones_image = face_recognition.load_image_file(img_dir + "jan_jones.png")
    jones_face_encoding = face_recognition.face_encodings(jones_image)[0]

    # Load a seventh sample picture and learn how to recognize it.
    williamshen_image = face_recognition.load_image_file(img_dir + "William_Shen.jpg")
    williamshen_face_encoding = face_recognition.face_encodings(williamshen_image)[0]

    # Load a eigth sample picture and learn how to recognize it.
    yusukewatanabe_image = face_recognition.load_image_file(img_dir + "Yusuke_Watanabe.jpg")
    yusukewatanabe_face_encoding = face_recognition.face_encodings(yusukewatanabe_image)[0]

    # Create arrays of known face encodings and their names
    known_face_encodings = [
        john_linss_face_encoding,
        jones_face_encoding,
        williamshen_face_encoding,
        yusukewatanabe_face_encoding
    ]
    known_face_names = {
        0: { "status": "See Ambassador", "jStatus": "アンバサダーを参照", "img": img_dir + "john_linss.jpg", "name": "John Linss", "username": "jlinss", "age": "21", "gender": "Male", "tier": "Platinum Member", "jTier": "プラチナ会員", "visits": "1,3", "residency": "Resident", "jResidency": "居住者", "prefix": "Mr."},
        1: { "status": "See Ambassador", "jStatus": "アンバサダーを参照", "img": img_dir + "jan_jones.png", "name": "Jan Jones Blackhurst", "username": "jblackhurst", "age": "21", "gender": "Female", "tier": "Seven Star Member", "jTier": "セブンスター会員", "visits": "1,2", "residency": "Non-resident", "jResidency": "非居住者", "prefix": "Mrs."},
        2: { "status": "See Ambassador", "jStatus": "アンバサダーを参照", "img": img_dir + "William_Shen.jpg", "name": "William Shen", "username": "wshen", "age": "28", "gender": "Male", "tier": "Platinum Member", "jTier": "プラチナ会員", "visits": "0,2", "residency": "Resident", "jResidency": "居住者", "prefix": "Mr."},
        3: { "status": "See Ambassador", "jStatus": "アンバサダーを参照", "img": img_dir + "Yusuke_Watanabe.jpg", "name": "Yusuke Watanabe", "username": "ywatanabe", "age": "26", "gender": "Male", "tier": "Seven Star Member", "jTier": "セブンスター会員", "visits": "1,3", "residency": "Resident", "jResidency": "居住者", "prefix": "Mr."}
    }
    unknown_face = { "name": "Unknown" }

    # Input Video Props
    scale_factor = 1

    # TODO: try different pre-trained models
    models = [
    "res10_300x300_ssd_iter_140000/"
    ]
    blob_sizes = [
    (300, 300)
    ]
    mean_val = [
    (104.0, 177.0, 123.0)
    ]
    model_index = 0 % len(models)
    model_version = models[model_index]
    
    prototxt = model_dir + model_version + "deploy.prototxt.txt"
    model = model_dir + model_version + "model.caffemodel"
    confidence = 0.50
    
    def __init__(self, socket="tcp://127.0.0.1", port=5559):
        multiprocessing.Process.__init__(self)
        self.stop_event = multiprocessing.Event()
        
        self.daemon = True
        self.socket = socket
        self.port = port
        # load our serialized model from disk
        print("[INFO] Loading CAFFE model...", end='\r')
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt, self.model)
        print("[INFO] Loading CAFFE model...done")
        
    def stop(self):
        self.stop_event.set()

    def recognize_face(self, crop_frame):

        if(crop_frame is None
            or crop_frame.shape[0]==0
            or crop_frame.shape[1]==0):
            return None
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = crop_frame[:, :, ::-1]

        startX = 0
        startY = 0
        endX = crop_frame.shape[1]
        endY = crop_frame.shape[0]
        
        left = startX
        top = startY
        right = endX
        bottom = endY

        face_locations = [(top, right, bottom, left)] # top right bottom left
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_encoding = face_encodings[0]
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
        distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
        face = self.unknown_face
        distance = 1

        # If a match was found in known_face_encodings, use the nearest one
        if True in matches:
            min_distance_index = np.argmin(distances)
            face = self.known_face_names[min_distance_index]
            distance = distances[min_distance_index]

        return face

    def process_frame_fast(self, frame, recognize=True):
        # Resize frame of video to 1/scale_factor size for faster face recognition processing
        frame = cv2.resize(frame, (0, 0), fx=float(1.0/self.scale_factor), fy=float(1.0/self.scale_factor))

        (h, w) = frame.shape[:2]
        blob_frame = cv2.resize(frame, self.blob_sizes[self.model_index])
        blob = cv2.dnn.blobFromImage(blob_frame, 1.0, self.blob_sizes[self.model_index], self.mean_val[self.model_index])
     
        # pass the blob through the network and obtain the
        # detections and predictions
        self.net.setInput(blob)
        detections = self.net.forward()
        faces = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence < self.confidence:
                continue

            # compute the (x, y)-coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (left, top, right, bottom) = box.astype("int")
            
            # crop the face from the original image
            crop_frame = frame[top:bottom, left:right]
            bounding_box = (left, top, right, bottom)

            face = self.unknown_face
            if(recognize == True):
                # perform face recognition on the cropped image
                face = self.recognize_face(crop_frame)
            
            # gather the metadata
            metadata = {}
            metadata['face'] = face
            metadata['confidence'] = str(confidence)
            metadata['left'] = str(left)
            metadata['top'] = str(top)
            metadata['right'] = str(right)
            metadata['bottom'] = str(bottom)
            
            faces.append(metadata)
            
        return faces

    def process_frame(self, frame, model="hog", recognize=True):
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=float(1.0/self.scale_factor), fy=float(1.0/self.scale_factor))

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        self.face_locations = face_recognition.face_locations(rgb_small_frame, model=model)
        self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
        faces = []
        for i in range(len(self.face_encodings)):
        
            (top, right, bottom, left) = self.face_locations[i]
            face_encoding = self.face_encodings[i]
            face = self.unknown_face
            confidence = 1
            
            if(recognize == True):
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                distance = 1

                # If a match was found in known_face_encodings, use the nearest one
                if True in matches:
                    min_distance_index = np.argmin(distances)
                    face = self.known_face_names[min_distance_index]
                    distance = distances[min_distance_index]
                    confidence = float(1.0/(distance+1.0))

            # gather the metadata
            metadata = {}
            metadata['face'] = face
            metadata['confidence'] = str(confidence)
            metadata['left'] = str(left)
            metadata['top'] = str(top)
            metadata['right'] = str(right)
            metadata['bottom'] = str(bottom)
            
            faces.append(metadata)
        return faces

    def start(self):
        while not self.stop_event.is_set():
            context = zmq.Context()
            
            # Initialize the Frame Processor Server
            frame_processor_server = context.socket(zmq.REP)
            frame_processor_server.bind(self.socket + ":" + str(self.port))
            #frame_processor_server.connect(self.socket + ":" + str(self.port))
            
            print("[INFO] Server Listening")

            # Loop and accept messages from both channels, acting accordingly
            while True:
                # receive the frame
                frame_metadata, frame = self.recv_frame(frame_processor_server)
                
                json_response = []
                # process the frame
                if(frame_metadata['fast'] == False):
                    json_response = self.process_frame(frame, model=frame_metadata['model'], recognize=frame_metadata['face_rec'])
                else:
                    json_response = self.process_frame_fast(frame, recognize=frame_metadata['face_rec'])
                
                # send the list of bounding boxes
                # and corresponding faces back
                frame_processor_server.send_json(json_response)

            if self.stop_event.is_set():
                frame_processor_server.close()
                context.term()
                break
    
    def recv_frame(self, frame_processor_server, flags=0, copy=True, track=False):
        frame_metadata = frame_processor_server.recv_json(flags=flags)
        frame_bytes = frame_processor_server.recv(flags=flags, copy=copy, track=track)
        
        # deserialize the incoming message
        frame = self.serde.deserialize(frame_bytes,
            dtype=frame_metadata['dtype'], shape=frame_metadata['shape'])
        return frame_metadata, frame

        
def main():
    print("[INFO] Initializing Frame Processor Server")
    #fps = FrameProcessorServer(socket="tcp://127.0.0.1", port=5559)
    #fps.start()
    """main function"""
    server = ServerTask(num_workers=4)
    server.start()
    #server.join()

if __name__ == "__main__":
    main()
