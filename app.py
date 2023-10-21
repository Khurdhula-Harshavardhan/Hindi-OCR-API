from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from hindi_ocr import CNN, RNN

app = Flask(__name__)
api = Api(app)

class OCRResource(Resource):
    def __init__(self):
        # Initialize models
        self.cnn_model = CNN()
        self.rnn_model = RNN()

    def post(self):
        # Extract base64 encoded image from request data
        try:
            data = request.get_json()
            base64_img = data.get("base64_image")
            if not base64_img:
                return jsonify({"Result": "Failed", "Error": "No base64_image provided."})

            # Predict using CNN and RNN
            cnn_prediction = self.cnn_model.extract_character(base64_img)
            rnn_prediction = self.rnn_model.extract_character(base64_img)

            # Return predictions
            return jsonify({"CNN": cnn_prediction, "RNN": rnn_prediction})

        except Exception as e:
            return jsonify({"Result": "Failed", "Error": str(e)})

api.add_resource(OCRResource, '/predict')

