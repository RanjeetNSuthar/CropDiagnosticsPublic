from flask import Flask
from flask import request
import firebase_admin
from firebase_admin import credentials, storage, firestore
from google.cloud import storage
from google.oauth2 import service_account
import pycountry_convert as pc
import recognizer
import detector
import os
import tensorflow as tf
import tensorflow_hub as hub


# Initialize the firebase App
cred = credentials.Certificate(
    "crop-diagnostic-firebase-adminsdk-5t8a2-4d3de59532.json")
firebase_admin.initialize_app(
    cred, {'storageBucket': 'crop-diagnostic.appspot.com'})

# Accessing service account credentials
credentials = service_account.Credentials.from_service_account_file(
    "crop-diagnostic-firebase-adminsdk-5t8a2-4d3de59532.json")


def country_to_continent(country_name):
    country_alpha2 = pc.country_name_to_country_alpha2(country_name)
    country_continent_code = pc.country_alpha2_to_continent_code(
        country_alpha2)
    country_continent_name = pc.convert_continent_code_to_continent_name(
        country_continent_code)
    return country_continent_name


db = firestore.client()

recognizer_model = tf.keras.models.load_model('recognizer.hdf5', custom_objects={
    'KerasLayer': hub.KerasLayer})

detector_model = hub.load(
    "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1").signatures['default']

app = Flask(__name__)


@app.route("/get_results", methods=['POST', 'GET'])
def get_results():
    response = {"isInsect": False,
                "pest category": None,
                "pesticide list": None}
    image_name = request.form['image name']
    storage.Client(credentials=credentials).bucket(firebase_admin.storage.bucket().name).blob(
        'images/'+image_name).download_to_filename('./images/'+image_name)
    image_path = "./images/"+image_name
    host_crop = request.form['crop'].lower()
    host_country = request.form['country'].lower()
    host_continent = country_to_continent(request.form['country'].title())
    categorical_features = [host_crop, host_country, host_continent]
    det_object = detector.Detector(image_path, detector_model)
    if det_object.run_detector():
        recg_object = recognizer.Recognizer(
            image_path, categorical_features, recognizer_model)
        pest_category = recg_object.Process_and_Predict()
        res = db.collection('pesticides').document(
            pest_category).get().to_dict()

        response['isInsect'] = True
        response['pest category'] = pest_category
        response['pesticide list'] = res

    os.remove('./images/'+image_name)
    return response


if __name__ == "__main__":
    app.run()
