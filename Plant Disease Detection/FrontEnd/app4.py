from flask import Flask, render_template, request
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt  # Import matplotlib for plotting

app = Flask(__name__)

# Load the pre-trained model
from keras.models import load_model
model = load_model('Mymodel1.h5')  # Replace 'model.h5' with the path to your trained model file

# Get the class indices and labels from the training dataset
class_dict = {
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy',
    4: 'Blueberry___healthy',
    5: 'Cherry_(including_sour)___Powdery_mildew',
    6: 'Cherry_(including_sour)___healthy',
    7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    8: 'Corn_(maize)___Common_rust_',
    9: 'Corn_(maize)___Northern_Leaf_Blight',
    10: 'Corn_(maize)___healthy',
    11: 'Grape___Black_rot',
    12: 'Grape___Esca_(Black_Measles)',
    13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    14: 'Grape___healthy',
    15: 'Orange___Haunglongbing_(Citrus_greening)',
    16: 'Peach___Bacterial_spot',
    17: 'Peach___healthy',
    18: 'Pepper,_bell___Bacterial_spot',
    19: 'Pepper,_bell___healthy',
    20: 'Potato___Early_blight',
    21: 'Potato___Late_blight',
    22: 'Potato___healthy',
    23: 'Raspberry___healthy',
    24: 'Soybean___healthy',
    25: 'Squash___Powdery_mildew',
    26: 'Strawberry___Leaf_scorch',
    27: 'Strawberry___healthy',
    28: 'Tomato___Bacterial_spot',
    29: 'Tomato___Early_blight',
    30: 'Tomato___Late_blight',
    31: 'Tomato___Leaf_Mold',
    32: 'Tomato___Septoria_leaf_spot',
    33: 'Tomato___Spider_mites Two-spotted_spider_mite',
    34: 'Tomato___Target_Spot',
    35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    36: 'Tomato___Tomato_mosaic_virus',
    37: 'Tomato___healthy'
}

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image file from the request
        imagefile = request.files['imagefile']

        # Save the uploaded image to a temporary location
        image_path = "uploaded_image.jpg"  # You can change this to any desired path
        imagefile.save(image_path)

        # Load and preprocess the image
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.  # Normalize pixel values

        # Perform prediction
        predictions = model.predict(img_array)

        # Decode the prediction
        predicted_class_index = np.argmax(predictions)
        
        # Get the list of class indices
        li = list(class_dict.keys())

        # Check if the predicted class index is within the range of class indices
        if predicted_class_index < len(li):
            predicted_class_label = class_dict[li[predicted_class_index]]
        else:
            predicted_class_label = "Unknown"

        # Plot the uploaded image with the predicted class label
        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.axis('off')
        plt.title(predicted_class_label)
        plt.show()

        # Pass the prediction result to the template
        return render_template('test2.html', prediction=predicted_class_label)

    # Render the form template for GET requests
    return render_template('test2.html')


if __name__ == '__main__':
    app.run(port=5000, debug=True)
