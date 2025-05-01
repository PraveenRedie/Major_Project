import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import time

# Set page configuration
st.set_page_config(
    page_title="Image Classification Ensemble",
    page_icon="🖼️",
    layout="wide"
)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Streamlit app states
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'class_names' not in st.session_state:
    st.session_state.class_names = []
if 'training_progress' not in st.session_state:
    st.session_state.training_progress = 0
if 'training_loss' not in st.session_state:
    st.session_state.training_loss = []

# Image preprocessing
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    return input_batch

# Create models
@st.cache_resource
def load_models(class_names):
    # ResNet model
    model_resnet = models.resnet18(pretrained=True)
    in_features = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(in_features, len(class_names))
    
    # MobileNet model
    model_mobilenet = models.mobilenet_v2(pretrained=True)
    in_features = model_mobilenet.classifier[1].in_features
    model_mobilenet.classifier[1] = nn.Linear(in_features, len(class_names))
    
    # DenseNet model
    model_densenet = models.densenet121(pretrained=True)
    in_features = model_densenet.classifier.in_features
    model_densenet.classifier = nn.Linear(in_features, len(class_names))
    
    # ConvNeXt model
    model_convnext = models.convnext_large(weights=models.ConvNeXt_Large_Weights.DEFAULT)
    in_features_convnext = model_convnext.classifier[2].in_features
    model_convnext.classifier[2] = nn.Linear(in_features_convnext, len(class_names))
    
    models_dict = {
        "ResNet": model_resnet,
        "MobileNet": model_mobilenet,
        "DenseNet": model_densenet,
        "ConvNeXt": model_convnext
    }
    
    # Load saved model weights if available
    for name, model in models_dict.items():
        model_path = f"{name.lower()}_model.pth"
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                st.success(f"Loaded {name} model weights successfully!")
            except Exception as e:
                st.warning(f"Could not load {name} model: {e}")
        else:
            st.info(f"{name} model weights not found, using default initialization.")
        model.to(device)
        model.eval()
    
    return models_dict

# Function to make predictions
def predict(image, models_dict, class_names):
    input_batch = preprocess_image(image)
    
    results = {}
    ensemble_prob_sum = None
    
    # Run inference on each model
    for model_name, model in models_dict.items():
        with torch.no_grad():
            outputs = model(input_batch)
            probs = torch.nn.functional.softmax(outputs, dim=1).squeeze()
            
            # Create a dictionary of class probabilities
            full_probs = {cls: float(prob) for cls, prob in zip(class_names, probs)}
            decided_idx = torch.argmax(probs).item()
            decided_class = class_names[decided_idx]
            decided_prob = float(probs[decided_idx])
            
            # Store results
            results[model_name] = {
                "full_probs": full_probs,
                "decided_class": decided_class,
                "decided_prob": decided_prob
            }
            
            # Accumulate probabilities for ensemble
            if ensemble_prob_sum is None:
                ensemble_prob_sum = probs.clone()
            else:
                ensemble_prob_sum += probs
    
    # Compute ensemble probabilities
    ensemble_probs = ensemble_prob_sum / len(models_dict)
    ensemble_full_probs = {cls: float(prob) for cls, prob in zip(class_names, ensemble_probs)}
    ensemble_decided_idx = torch.argmax(ensemble_probs).item()
    ensemble_decided_class = class_names[ensemble_decided_idx]
    ensemble_decided_prob = float(ensemble_probs[ensemble_decided_idx])
    
    # Store ensemble results
    results["Ensemble"] = {
        "full_probs": ensemble_full_probs,
        "decided_class": ensemble_decided_class,
        "decided_prob": ensemble_decided_prob
    }
    
    return results

# Function to plot probability distribution
def plot_probabilities(probs, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    classes = list(probs.keys())
    probabilities = list(probs.values())
    
    # Sort by probability if there are many classes
    if len(classes) > 5:
        sorted_indices = np.argsort(probabilities)[::-1][:5]  # Top 5 classes
        classes = [classes[i] for i in sorted_indices]
        probabilities = [probabilities[i] for i in sorted_indices]
    
    sns.barplot(x=probabilities, y=classes, ax=ax)
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.set_xlabel('Probability')
    ax.set_ylabel('Class')
    
    # Convert plot to image
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    """Train the model and return training history."""
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            # Update session state for progress bar
            st.session_state.training_progress += 1
            
        train_loss = running_loss / len(train_loader.dataset)
        history["train_loss"].append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        # Update session state for display
        st.session_state.training_loss.append((train_loss, val_loss, val_acc))
        
    return history

# Plot training history
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training and validation loss
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], 'bo-', label='Training Loss')
    ax1.plot(epochs, history["val_loss"], 'ro-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot validation accuracy
    ax2.plot(epochs, history["val_acc"], 'go-', label='Validation Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0, 1])
    ax2.legend()
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

# Function to load dataset
def load_dataset(data_dir):
    # Define transformations
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        st.error(f"Training or validation directory not found in {data_dir}")
        return None, None, []
    
    try:
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
        
        # Get class names
        class_names = train_dataset.classes
        
        # Create data loaders
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        return train_loader, val_loader, class_names
    
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None, None, []

# Main app function
def main():
    st.title("Image Classification Ensemble")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Predict", "Train Models", "About"])
    
    if app_mode == "Predict":
        prediction_page()
    elif app_mode == "Train Models":
        training_page()
    else:
        about_page()

# Prediction page
def prediction_page():
    st.header("Image Classification")
    st.write("Upload an image or take one with your webcam to classify it.")
    
    # Get class names
    class_names_input = st.text_input(
        "Enter class names (comma-separated)", 
        value=",".join(st.session_state.class_names) if st.session_state.class_names else ""
    )
    
    if class_names_input:
        st.session_state.class_names = [name.strip() for name in class_names_input.split(",")]
    
    if not st.session_state.class_names:
        st.warning("Please enter class names to continue.")
        return
    
    # Load models
    if not st.session_state.models_loaded:
        with st.spinner("Loading models..."):
            st.session_state.models = load_models(st.session_state.class_names)
            st.session_state.models_loaded = True
    
    # Input methods (file uploader and webcam)
    st.sidebar.title("Input Options")
    input_method = st.sidebar.radio("Select input method:", ["Upload Image", "Use Webcam"])
    
    image = None
    
    if input_method == "Upload Image":
        uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
    else:  # Use Webcam
        picture = st.camera_input("Take a picture")
        if picture is not None:
            image = Image.open(picture).convert('RGB')
    
    # Process image and display results
    if image is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image, caption="Input Image", use_column_width=True)
        
        with col2:
            with st.spinner("Making predictions..."):
                results = predict(image, st.session_state.models, st.session_state.class_names)
                
                # Display prediction results
                st.subheader("Prediction Results")
                
                # Create tabs for each model and ensemble
                model_tabs = st.tabs(list(results.keys()))
                
                for i, (model_name, model_tab) in enumerate(zip(results.keys(), model_tabs)):
                    with model_tab:
                        result = results[model_name]
                        
                        # Display the predicted class and confidence
                        st.markdown(f"**Predicted Class:** {result['decided_class']}")
                        st.markdown(f"**Confidence:** {result['decided_prob']:.4f} ({result['decided_prob']*100:.2f}%)")
                        
                        # Display probability distribution
                        prob_buf = plot_probabilities(result["full_probs"], f"{model_name} Probability Distribution")
                        st.image(prob_buf)

# Training page
def training_page():
    st.header("Model Training")
    st.write("Train the ensemble models on your own dataset.")
    
    # Data directory input
    data_dir = st.text_input("Enter the path to your dataset directory", "")
    
    if not data_dir:
        st.info("Please enter the path to your dataset directory.")
        st.write("""
        The dataset directory should have the following structure:
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
        """)
        return
    
    # Try to load dataset
    with st.spinner("Loading dataset..."):
        train_loader, val_loader, class_names = load_dataset(data_dir)
    
    if not train_loader or not val_loader:
        return
    
    st.success(f"Dataset loaded successfully! Found {len(class_names)} classes: {', '.join(class_names)}")
    st.session_state.class_names = class_names
    
    # Training parameters
    st.subheader("Training Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
        batch_size = st.number_input("Batch Size", min_value=1, max_value=128, value=32)
        
    with col2:
        num_epochs = st.number_input("Number of Epochs", min_value=1, max_value=100, value=5)
        model_to_train = st.selectbox("Model to Train", ["ResNet", "MobileNet", "DenseNet", "ConvNeXt", "All"])
    
    # Start training
    if st.button("Start Training"):
        if not st.session_state.models_loaded:
            with st.spinner("Loading models..."):
                st.session_state.models = load_models(class_names)
                st.session_state.models_loaded = True
        
        # Reset training progress
        st.session_state.training_progress = 0
        st.session_state.training_loss = []
        
        # Calculate total steps for progress bar
        total_steps = len(train_loader) * num_epochs
        progress_bar = st.progress(0)
        
        # Create placeholders for displaying loss and accuracy
        loss_placeholder = st.empty()
        history_placeholder = st.empty()
        
        # Define loss function
        criterion = nn.CrossEntropyLoss()
        
        models_to_train = []
        if model_to_train == "All":
            models_to_train = list(st.session_state.models.keys())
        else:
            models_to_train = [model_to_train]
            
        # Train selected models
        for model_name in models_to_train:
            st.subheader(f"Training {model_name}...")
            model = st.session_state.models[model_name]
            
            # Create optimizer
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Train the model
            start_time = time.time()
            history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)
            training_time = time.time() - start_time
            
            # Update progress bar
            progress_percentage = st.session_state.training_progress / (total_steps * len(models_to_train))
            progress_bar.progress(min(progress_percentage, 1.0))
            
            # Display training metrics
            loss_placeholder.write(f"Latest metrics - Train Loss: {history['train_loss'][-1]:.4f}, "
                                  f"Val Loss: {history['val_loss'][-1]:.4f}, "
                                  f"Val Accuracy: {history['val_acc'][-1]:.4f}")
            
            # Display training history plot
            history_plot = plot_training_history(history)
            history_placeholder.image(history_plot)
            
            # Save model
            torch.save(model.state_dict(), f"{model_name.lower()}_model.pth")
            st.success(f"{model_name} trained successfully in {training_time:.2f} seconds!")
            
        st.success("All selected models have been trained and saved!")

# About page
def about_page():
    st.header("About the Image Classification Ensemble")
    
    st.write("""
    ## Overview
    This application uses an ensemble of convolutional neural networks to classify images. The ensemble approach combines 
    predictions from multiple models to achieve better accuracy and robustness.
    
    ## Models Used
    This application uses the following pre-trained models:
    
    - **ResNet18**: A residual network with 18 layers.
    - **MobileNetV2**: A lightweight model designed for mobile and edge devices.
    - **DenseNet121**: A densely connected convolutional network.
    - **ConvNeXt Large**: A modern ConvNet architecture.
    
    The ensemble combines predictions from all models by averaging their softmax probabilities.
    
    ## How to Use
    
    ### Prediction
    1. Navigate to the "Predict" tab
    2. Enter your class names (comma-separated)
    3. Upload an image or take a picture with your webcam
    4. View the prediction results from each model and the ensemble
    
    ### Training
    1. Navigate to the "Train Models" tab
    2. Enter the path to your dataset directory
    3. Adjust training parameters as needed
    4. Click "Start Training" to fine-tune the models on your data
    
    ## Dataset Format
    Your dataset should be organized in the following structure:
    ```
    data_dir/
    ├── train/
    │   ├── class1/
    │   │   ├── image1.jpg
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
    """)

if __name__ == "__main__":
    main()