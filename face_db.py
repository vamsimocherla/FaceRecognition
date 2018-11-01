import face_recognition

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

# Load a eigth sample picture and learn how to recognize it.
krishna_image = face_recognition.load_image_file(img_dir + "krishna.jpg")
krishna_face_encoding = face_recognition.face_encodings(krishna_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    john_linss_face_encoding,
    jones_face_encoding,
    williamshen_face_encoding,
    yusukewatanabe_face_encoding,
    krishna_face_encoding
]
known_face_names = {
    0: {"status": "See Ambassador", "jStatus": "アンバサダーを参照", "img": img_dir + "john_linss.jpg", "name": "John Linss",
        "username": "jlinss", "age": "21", "gender": "Male", "tier": "Platinum Member", "jTier": "プラチナ会員",
        "visits": "1,3", "residency": "Resident", "jResidency": "居住者", "prefix": "Mr."},
    1: {"status": "See Ambassador", "jStatus": "アンバサダーを参照", "img": img_dir + "jan_jones.png",
        "name": "Jan Jones Blackhurst", "username": "jblackhurst", "age": "21", "gender": "Female",
        "tier": "Seven Star Member", "jTier": "セブンスター会員", "visits": "1,2", "residency": "Non-resident",
        "jResidency": "非居住者", "prefix": "Mrs."},
    2: {"status": "See Ambassador", "jStatus": "アンバサダーを参照", "img": img_dir + "William_Shen.jpg",
        "name": "William Shen", "username": "wshen", "age": "28", "gender": "Male", "tier": "Platinum Member",
        "jTier": "プラチナ会員", "visits": "0,2", "residency": "Resident", "jResidency": "居住者", "prefix": "Mr."},
    3: {"status": "See Ambassador", "jStatus": "アンバサダーを参照", "img": img_dir + "Yusuke_Watanabe.jpg",
        "name": "Yusuke Watanabe", "username": "ywatanabe", "age": "26", "gender": "Male",
        "tier": "Seven Star Member", "jTier": "セブンスター会員", "visits": "1,3", "residency": "Resident",
        "jResidency": "居住者", "prefix": "Mr."},
    4: {"status": "See Ambassador", "jStatus": "アンバサダーを参照", "img": img_dir + "krishna.jpg",
       "name": "Krishna", "username": "kmocherla", "age": "29", "gender": "Male",
       "tier": "Seven Star Member", "jTier": "セブンスター会員", "visits": "1,3", "residency": "Non-Resident",
       "jResidency": "居住者", "prefix": "Mr."}
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
