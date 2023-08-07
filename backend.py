import os
from flask import Flask, request, jsonify
import dlib
import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
import hashlib
import numpy as np

app = Flask(__name__)

# Initialize the face detection and recognition models
face_detector = dlib.get_frontal_face_detector()
face_recognizer = dlib.face_recognition_model_v1('dlib_models/dlib_face_recognition_resnet_model_v1.dat')

# Directory to store images
IMAGES_DIR = 'images'

# Function to process the uploaded image
def process_image(image):
    img = Image.open(image)
    img_gray = img.convert('L')
    img_np = np.array(img_gray)  # Convert PIL Image to numpy array
    dets = face_detector(img_np, 1)  # Use the numpy array for face detection

    if len(dets) == 0:
        return None

    # Assume only one face in the image
    shape = face_recognizer(img_np, dets[0])
    face_descriptor = face_recognizer.compute_face_descriptor(img_np, shape)
    return face_descriptor

# Function to download images from URLs
def download_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        return BytesIO(response.content)
    return None

# Function to search for images using a search engine
def search_images(name):
    search_url = f'https://www.example.com/images?q={name}'  # Replace with your search engine URL
    response = requests.get(search_url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        image_urls = []

        # Find image URLs from the search results page
        for img_tag in soup.find_all('img'):
            image_url = img_tag.get('src')
            if image_url and image_url.startswith('http'):
                image_urls.append(image_url)

        return image_urls

    return []

@app.route('/analyze', methods=['POST'])
def analyze_face():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    image = request.files['image']
    face_descriptor = process_image(image)

    if face_descriptor is None:
        return jsonify({'error': 'No face detected'})

    # Generate a unique filename using the face descriptor
    file_hash = hashlib.md5(str(face_descriptor).encode('utf-8')).hexdigest()
    filename = f"{file_hash}.jpg"

    # Save the image locally
    image.save(os.path.join(IMAGES_DIR, filename))

    # Perform web scraping to search for other images of the recognized person
    search_name = 'Claire McIntyre'  # Replace with the recognized person's name
    image_urls = search_images(search_name)

    # Download and save the additional images
    for idx, image_url in enumerate(image_urls):
        image_data = download_image(image_url)
        if image_data:
            additional_filename = f"{file_hash}_{idx}.jpg"
            with open(os.path.join(IMAGES_DIR, additional_filename), 'wb') as f:
                f.write(image_data.getvalue())

    return jsonify({'success': 'Face analyzed and images saved'})

if __name__ == '__main__':
    # Create the IMAGES_DIR if it doesn't exist
    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)

    app.run(debug=True)
