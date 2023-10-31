from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from hindi_ocr import CNN, RNN
from flask_cors import CORS
import numpy as np
import base64

app = Flask(__name__)
api = Api(app)
CORS(app)

def make_serializable(data):
    """Converts numpy arrays and numpy data types to python native types."""
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, (np.float32, np.int32, np.int64)):
        return float(data)
    if isinstance(data, dict):
        return {k: make_serializable(v) for k, v in data.items()}
    return data

class OCRResource(Resource):
    # Predict using CNN and RNN
    cnn_model = CNN()
    rnn_model = RNN()
    def post(self):
        # Extract base64 encoded image from request data
        try:
            data = request.get_json()
            base64_img = data.get("base64_image")
            if not base64_img:
                return jsonify({"Result": "Failed", "Error": "No base64_image provided."})

            
            cnn_prediction = self.cnn_model.extract_character(base64_img)
            rnn_prediction = self.rnn_model.extract_character(base64_img)

            # Make predictions serializable
            cnn_prediction = make_serializable(cnn_prediction)
            rnn_prediction = make_serializable(rnn_prediction)
            # Return predictions
            return jsonify({"CNN": cnn_prediction}, {"RNN": rnn_prediction})

        except Exception as e:
            return jsonify({"Result": "Failed", "Error": str(e)})

api.add_resource(OCRResource, '/predict')

