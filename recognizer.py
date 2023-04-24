import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


class Recognizer:
    def __init__(self, img_path, categorical_features, recognizer):
        self.img_path = img_path
        self.categorical_features = categorical_features
        self.recognizer = recognizer

    def process_image(self, image_path, size=224):
        '''
        takes image file path as an input and returns a tensor of specified size = (size x size)
        '''

        image = tf.io.read_file(image_path)
        # Turn the jpeg tensor image into numerical Tensor with 3 colour channels (Red, Green, Blue)
        image = tf.image.decode_image(
            image, channels=3, expand_animations=False)
        # convert the colour channel value from (0-255) to (0-1)
        image = tf.image.convert_image_dtype(image, tf.float32)
        # Resize image as (size X size)
        image = tf.image.resize(image, size=[size, size])

        return image

    def Process_and_Predict(self):
        # trained_model = tf.keras.models.load_model('recognizer.hdf5', custom_objects={
        #                                            'KerasLayer': hub.KerasLayer})
        pest_type = pd.read_csv("category_chart.csv")["type"]

        Deployment_Categorical_Input = pd.read_csv(
            "Deployment_Categorical_Input.csv")
        Deployment_Categorical_Input.loc[0][self.categorical_features[0]] = 1
        Deployment_Categorical_Input.loc[0][self.categorical_features[1]] = 1
        Deployment_Categorical_Input.loc[0][self.categorical_features[2]] = 1
        categorical_ds = tf.data.Dataset.from_tensor_slices(
            Deployment_Categorical_Input)

        image_df = pd.DataFrame(data=[self.img_path], columns=["image"])
        image_df = image_df['image'].to_numpy()
        image_ds = tf.data.Dataset.from_tensor_slices(image_df)
        image_ds = image_ds.map(self.process_image)

        test_input = tf.data.Dataset.zip(
            {"image_in": image_ds, "categorical_in": categorical_ds}).batch(32)
        predictions = self.recognizer.predict(test_input, verbose=1)

        return pest_type[np.argmax(predictions)]
