import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
import os

# Get the current working directory
current_directory = os.getcwd()

# Construct the relative path to the model file
model_path = os.path.join(current_directory, 'model.h5')

# Load the trained model
model = load_model(model_path)

# Function to preprocess and predict the uploaded image
def predict_image():
    file_path = filedialog.askopenfilename()  # Open file dialog to select an image
    if file_path:
        image = Image.open(file_path)  # Open the selected image using PIL
        image = image.resize((150, 150))  # Resize the image to match model input size
        image = np.array(image) / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        prediction = model.predict(image)  # Make prediction
        predicted_class = np.argmax(prediction, axis=1)[0]  # Get the predicted class index
        class_names = ['Aster', 'Daisy', 'Iris', 'Lavender', 'Lily', 'Marigold', 'Orchid', 'Poppy', 'Rose', 'Sunflower']
        predicted_label = class_names[predicted_class]  # Get the class label from index

        # Display the predicted class in the label
        result_label.config(text=f"Predicted class: {predicted_label}")

# Create main window
root = tk.Tk()
root.title('Flower Classification App')

# Create a button to upload an image for prediction
upload_button = tk.Button(root, text='Upload Image', command=predict_image)
upload_button.pack()

# Create a label to display prediction result
result_label = tk.Label(root, text='Prediction will appear here')
result_label.pack()

# Run the GUI main loop
root.mainloop()
