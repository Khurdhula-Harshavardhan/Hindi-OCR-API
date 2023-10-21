"""
Module Description: This module serves an API, for using CNN and a RNN.
"""
from joblib import load
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import json
from tensorflow.keras.preprocessing import image

class Image_Processor():
    """
    This Module acts as a middleware between the api and the model.
    Specifically the models should use this class to read image information,
    Get binary image data which is needed for their processing.

    Attributes:

        raw_image_data (base64): Has encoded image information.
        image_data (bytes): Stores decoded bytes.
        image (np.array): Stores the image in np array form.
    """
    raw_image_data = None
    image_data = None
    image = None

    def read_image(self, image_data: bytes) -> float:
        """
        Image_Processor.read_image(): takes in the encoded image data and then converts into an numpy array for further processing of the image.
        """
        try:
            self.raw_image_data = image_data
            self.image_data = base64.b64decode(self.raw_image_data)
            self.image = image.img_to_array(image.load_img(BytesIO(self.image_data), target_size=(32, 32)))
            

            return self.image
        except Exception as e:
            return {"Result": "Failed",
                    "ERROR": str(e),
                    "ERR at": "hindi_ocr.Image_Processor.read_image"}
        
    def read_grayscale_image(self, image_data: bytes) -> float:
        """
        Image_Processor.read_image(): takes in the encoded image data and then converts into an numpy array for further processing of the image.
        """
        try:
            self.raw_image_data = image_data
            self.image_data = base64.b64decode(self.raw_image_data)
            self.image = image.img_to_array(image.load_img(BytesIO(self.image_data), target_size=(32, 32), color_mode="grayscale"))
            

            return self.image
        except Exception as e:
            return {"Result": "Failed",
                    "ERROR": str(e),
                    "ERR at": "hindi_ocr.Image_Processor.read_grayscale_image"}

class CNN(Image_Processor):
    """
    The CNN class is the API for the previously trained model. Which uses CNN.joblib that was trained previously by @Harsha Vardhan Khurdula.
    """
    cnn = None
    base_image_data = None
    image = None
    labels = None
    original_characters = None
    confidences = None

    def __init__(self) -> None:
        """
        The constructor should initialize the model.
        """
        try:
            self.cnn = load("Models/CNN.joblib")
            self.labels = json.load(open('Models/cnn_labels.json', 'r'))
            self.original_characters = json.load(open('Models/labels.json', 'r', encoding="utf-8"))
            
            if self.cnn is None:
                raise FileNotFoundError("Cannot load pretrained CNN, please check /Models/ to verify if the model does exist.")
            else:
                print("[SUCCESS] The pretrained CNN model has been loaded successfully.")
        except FileNotFoundError as e:
            print("[ERROR] The following error occured while trying to load the model: "+str(e))

    def get_confidences(self, predictions: list) -> dict:
        """
        The get_confidences method aims to build a dictionary giving key: value pairs of characters and their confidences for a classification by CNN.
        """
        try:
            self.confidences = dict()
            for i in range(len(predictions)):
                confidence = predictions[i]
                hindi_char = self.original_characters.get(self.labels.get(str(i)))
                self.confidences[hindi_char] = confidence
            return self.confidences
        except Exception as e:
            return {"Result": "Failed",
                    "ERROR": str(e),
                    "ERR at": "hindi_ocr.CNN.get_confindences"}

    def extract_character(self, image: bytes) -> dict:
        """
        Extract character is a CNN method that should accept bytes image data and then makes prediction for it. 
        """
        try:
            self.base_image_data = image
            self.image = self.read_image(image_data= self.base_image_data)
            self.image = self._preprocess_image(self.image)
            predictions = self.cnn.predict(self.image)
            predicted_class = np.argmax(predictions[0])
            confidences = self.get_confidences(predictions=predictions[0])
            return {
                "Result": "Success",
                "Model": "CNN",
                "Type": "Sequential",
                "Prediction": self.original_characters.get(self.labels.get(str(predicted_class), "NaN")),
                "Confidence": predictions[0][predicted_class],
                "Other Confidences": confidences
            }
        except Exception as e:
            return {"Result": "Failed",
                    "ERROR": str(e),
                    "ERR at": "hindi_ocr.CNN.extract_character"}
        
    def _preprocess_image(self, img: np.ndarray)-> np.ndarray:
        """
        Predict_with_Preprocessing is an private method that preprocesses image for classification
        """
        try:

            img_array = image.img_to_array(img)
            img_array /= 255.
            img_array_reshaped = np.expand_dims(img_array, axis=0)
            
            
            return img_array_reshaped
        except Exception as e:
            return {"Result": "Failed",
                    "ERROR": str(e),
                    "ERR at": "hindi_ocr.CNN.(PRIVATE_METHOD)"}


class RNN(Image_Processor):
    """
    The RNN class is the API for the previously trained model. Which uses RNN.joblib that was trained previously by @Harsha Vardhan Khurdula.
    """
    rnn = None
    base_image_data = None
    image = None
    labels = None
    original_characters = None
    confidences = None
    def __init__(self) -> None:
        """
        The constructor should initialize the model.
        """
        try:
            self.rnn = load("Models/RNN.joblib")
            self.labels = json.load(open('Models/rnn_labels.json', 'r'))
            self.original_characters = json.load(open('Models/labels.json', 'r', encoding="utf-8"))
            if self.rnn is None:   
                raise FileNotFoundError("Cannot load pretrained RNN, please check /Models/ to verify if the model does exist.")
            else:
                print("[SUCCESS] The pretrained RNN model has been loaded successfully.")
        except FileNotFoundError as e:
            print("[ERROR] The following error occured while trying to load the model: "+str(e))

    def get_confidences(self, predictions: list) -> dict:
        """
        The get_confidences method aims to build a dictionary giving key: value pairs of characters and their confidences for a classification by RNN.
        """
        try:
            self.confidences = dict()
            for i in range(len(predictions)):
                confidence = predictions[i]
                hindi_char = self.original_characters.get(self.labels.get(str(i)))
                self.confidences[hindi_char] = confidence
            return self.confidences
        except Exception as e:
            return {"Result": "Failed",
                    "ERROR": str(e),
                    "ERR at": "hindi_ocr.RNN.get_confindences"}

    def extract_character(self, image: bytes) -> dict:
        """
        Extract character is a RNN method that should accept bytes image data and then makes prediction for it. 
        """
        try:
            self.base_image_data = image
            self.image = self.read_grayscale_image(image_data= self.base_image_data)
            self.image = self._preprocess_image(self.image)
            predictions = self.rnn.predict(self.image)
            predicted_class = np.argmax(predictions[0])
            confidences = self.get_confidences(predictions=predictions[0])
            return {
                "Result": "Success",
                "Model": "RNN",
                "Type": "Sequential",
                "Prediction": self.original_characters.get(self.labels.get(str(predicted_class), "NaN")),
                "Confidence": predictions[0][predicted_class],
                "Other Confidences": confidences
            }
        except Exception as e:
            return {"Result": "Failed",
                    "ERROR": str(e),
                    "ERR at": "hindi_ocr.RNN.extract_character"}
        
    def _preprocess_image(self, img: np.ndarray)-> np.ndarray:
        """
        Predict_with_Preprocessing is an private method that preprocesses image for classification
        """
        try:

            img_array = image.img_to_array(img)
            img_array_reshaped = np.expand_dims(img_array, axis=0)
            img_array_normalized = img_array_reshaped / 255.0
            
            
            return img_array_normalized
        except Exception as e:
            return {"Result": "Failed",
                    "ERROR": str(e),
                    "ERR at": "hindi_ocr.RNN.(PRIVATE_METHOD)"}
        