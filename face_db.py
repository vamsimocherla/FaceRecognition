import face_recognition
import os

# get the current working directory
cwd = os.path.dirname(os.path.realpath(__file__))

# Image DB location
img_dir = cwd + "/static/img/"
model_dir = cwd + "/static/models/"

# Load a sample picture and learn how to recognize it.
sample_image = face_recognition.load_image_file(img_dir + "barack.jpg")
sample_face_encoding = face_recognition.face_encodings(sample_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    sample_face_encoding
]
known_face_names = {
    0: {"img": img_dir + "barack.jpg", "name": "Barack Obama",
        "username": "bobama", "age": "21", "gender": "Male"}
}
unknown_face = {"name": "Unknown"}

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
confidence = 0.70
