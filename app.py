from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('best_model.keras')

# Error Level Analysis func
def convert_to_ela_image(path, quality):
    temp_filename = 'temp_file_name.jpg'

    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = Image.open(temp_filename)

    ela_image = ImageChops.difference(image, temp_image)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff

    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    os.remove(temp_filename)

    return ela_image

# Gamma Correction func
def apply_gamma_correction(image, gamma=1.75):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Prepare image function
# Prepare image function
def prepare_image(image_path):
    quality = 90
    image_size = (224, 224)

    # Apply ELA
    ela_image = convert_to_ela_image(image_path, quality)

    # Apply gamma correction on ELA image
    ela_image_corrected = apply_gamma_correction(np.array(ela_image))

    # Resize implemented image to the target size
    ela_image_resized = cv2.resize(ela_image_corrected, image_size)

    processed_image_path = 'static/processed_image.jpg'
    cv2.imwrite(processed_image_path, ela_image_resized)

    return ela_image_resized / 255.0


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict_page():
    return render_template('predict.html')

@app.route('/about', methods=['GET'])
def about_page():
    return render_template('about.html')

@app.route('/result', methods=['POST'])
def result():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        try:
            # Save the file
            filename2 = 'static/uploaded_image.jpg'
            file.save(filename2)

            # Prepare the image
            prepared_image = prepare_image(filename2)

            # Make prediction
            prediction = model.predict(np.expand_dims(prepared_image, axis=0))
            #prediction_percentage = prediction[0][0] * 100
            prediction_percentage_0 = prediction[0][0] * 100
            prediction_percentage_1 = (1 - prediction[0][0]) * 100

            # Format the prediction result
            if prediction[0][0] > 0.5:
                result = "Tampered"
            else:
                result = "Authentic"

            #return render_template('result.html', result=result, prediction_percentage=prediction_percentage)
            return render_template('result.html', result=result, prediction_percentage_0=prediction_percentage_0, prediction_percentage_1=prediction_percentage_1)
        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
