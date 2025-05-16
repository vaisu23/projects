# Mathematical Symbol Seeker

This is an academic mini-project aimed at building a Mathematical Symbol Seeker — a system that can recognize handwritten mathematical Greek symbols and classify them correctly using a Convolutional Neural Network (CNN).

## Project Description

The goal of this project is to create an AI-powered tool that:

Takes handwritten input of a mathematical symbol (e.g θ, σ, α, etc.)


Processes the input image to ensure uniformity

Uses a CNN model to predict which mathematical symbol was written

This tool is useful in building intelligent systems for digitizing mathematical notes, aiding accessibility tools, and educational software.


### Requirements
Make sure the following Python libraries are installed:

     pip install numpy matplotlib tensorflow pillow


 ### Usage
Follow these steps to run the handwritten mathematical symbol classification system:

1. Prepare Your Dataset
Place your handwritten symbol images in the data/ directory.
Make sure the images are in a standard image format like .png, .jpg, or .jpeg.

Each image should contain one handwritten mathematical symbol, such as θ, σ, α, etc.

2. Preprocess the Images
Run the preprocessing script to convert raw images into a consistent format suitable for model training.

        python conversion.py
 
What this does:

Converts images to black and white (grayscale or binary)

Resizes them to a uniform size

Normalizes image data for input into the CNN

This step ensures that all images are in the same shape and format before feeding them into the model.

3. Train and Test the CNN Model
Now, train the Convolutional Neural Network and test its performance:

        python test22.py
What this does:

Loads the preprocessed dataset

Batches and feeds the data into the CNN

Trains the model to recognize different symbol classes

Tests the trained model on new images and prints the predicted symbol

### Team Members
VAISAKH

INDRAJITH

ARFATH

ARJUN










