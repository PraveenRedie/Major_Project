# Image Classification Ensemble with Streamlit

This repository contains a Streamlit web application that uses an ensemble of deep learning models for image classification. The ensemble combines predictions from ResNet, MobileNet, DenseNet, and ConvNeXt models to provide robust classification results.

## Features

- Image classification using pre-trained models
- Ensemble prediction by averaging model outputs
- Interactive web interface with Streamlit
- Option to upload images or take photos with webcam
- Model training capability with your own dataset
- Visualization of prediction probabilities and training metrics

## Installation

1. Clone this repository or download the files

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Basic Application

1. Run the basic Streamlit app:
   ```
   streamlit run app.py
   ```

2. Access the application in your web browser (usually at http://localhost:8501)

3. Enter your class names (comma-separated list)

4. Upload an image or take a photo with your webcam

5. View the classification results from each model and the ensemble

### Advanced Application

1. Run the advanced Streamlit app:
   ```
   streamlit run advanced_app.py
   ```

2. The advanced application includes:
   - Model training capability
   - Interactive visualization of training metrics
   - Multiple pages for different functions

### Training on Your Own Dataset

To train the models on your own dataset:

1. Organize your dataset in the following structure:
   ```
   data_dir/
   ├── train/
   │   ├── class1/
   │   │   ├── image1.jpg
   │   │   ├── image2.jpg
   │   │   └── ...
   │   ├── class2/
   │   │   └── ...
   │   └── ...
   └── val/
       ├── class1/
       │   └── ...
       ├── class2/
       │   └── ...
       └── ...
   ```

2. Navigate to the "Train Models" tab in the advanced application

3. Enter the path to your dataset directory

4. Configure training parameters (learning rate, batch size, epochs)

5. Select which model(s) to train

6. Click "Start Training"

## Pre-trained Model Weights

If you have trained models in the original Python script, you can use those weights with this application:

1. Make sure your model weights files (resnet_model.pth, mobilenet_model.pth, densenet_model.pth, convnext_model.pth) are in the same directory as the Streamlit app

2. The application will automatically load these weights if they exist

## Adapting to Your Specific Classification Task

1. Before running the app, update the class names to match your specific categories:
   - In app.py: Update the `class_names` list with your categories
   - In model_loader.py: Make sure the `class_names` match your categories if you're generating new model weights

2. If you need to adjust the model architecture to match your original training:
   - Check that the final classification layer dimensions match your number of classes
   - Ensure any normalization or preprocessing steps match your original training setup

## Troubleshooting

### Missing Model Weights

If you get an error about missing model weights:

1. Either run the training process first, or
2. Use the model_loader.py script to create initial model weights:
   ```
   python model_loader.py
   ```

### CUDA Out of Memory

If you encounter CUDA out of memory errors:

1. Reduce the batch size
2. Use a smaller model (MobileNet is the most efficient)
3. Set `device = torch.device("cpu")` to force CPU computation

### Wrong Predictions

If predictions seem incorrect:

1. Verify that class names match your original training
2. Ensure image preprocessing matches the original training
3. Check that model weights are loaded correctly

## Deployment

You can deploy this Streamlit app to various platforms:

- [Streamlit Cloud](https://streamlit.io/cloud)
- [Heroku](https://heroku.com)
- [AWS](https://aws.amazon.com)
- [Microsoft Azure](https://azure.microsoft.com)

For deployment, ensure that your requirements.txt file includes all dependencies, and adjust file paths to be relative to the application directory.
