import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
import seaborn as sns

# Load the trained model
try:
    MODEL_FILEPATH = 'Trained Models/dstsmcgselnn_mnist.keras'
    model = tf.keras.models.load_model(MODEL_FILEPATH)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: The {MODEL_FILEPATH} model file was not found.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the model: {str(e)}")
    exit()

def preprocess_image(image_path):
    try:
        # Load and preprocess an image using OpenCV
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error: Image at path '{image_path}' could not be loaded.")
            return None
        image = cv2.resize(image, (28, 28))
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=[0, -1])  # Shape to (1, 28, 28, 1)
        return image
    except Exception as e:
        print(f"An error occurred during image preprocessing: {str(e)}")
        return None

def predict(image_path):
    image = preprocess_image(image_path)
    if image is None:
        print("Skipping prediction due to preprocessing error.")
        return None
    
    try:
        predictions = model.predict(image)
        # The first output is the classification result
        classification_output = predictions[0]
        predicted_digit = np.argmax(classification_output)
        confidence = np.max(classification_output)
        
        print(f"Prediction shape: {[p.shape for p in predictions]}")
        print(f"Prediction types: {[type(p) for p in predictions]}")
        
        return predicted_digit, confidence
    except Exception as e:
        print(f"An error occurred during prediction: {str(e)}")
        return None

def display_image_and_prediction(image_path, predicted_digit, confidence):
    # Display the image along with its predicted digit and confidence
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Image at path '{image_path}' could not be loaded.")
        return
    
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.title(f'Predicted digit: {predicted_digit}\nConfidence: {confidence:.2f}')
    plt.axis('off')
    plt.show()

def evaluate_model_on_test_data(x_test, y_test):
    """
    This function evaluates the model on the test data and prints various metrics such as:
    F1-score, confusion matrix, precision, recall, and prevalence.
    """
    # Get model predictions on the test set
    y_pred_probs = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred_probs[0], axis=1)  # The first output corresponds to classification
    y_true_classes = np.argmax(y_test, axis=1)  # Convert one-hot encoded labels to class indices
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Precision, Recall, F1-score
    precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
    recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
    f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

    # Prevalence is the proportion of actual positives (true samples of each class)
    prevalence_per_class = np.bincount(y_true_classes) / len(y_true_classes)

    print(f"Classification Report:\n{classification_report(y_true_classes, y_pred_classes)}")
    print(f"Precision (Specificity): {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    # Output prevalence per class
    for i, prevalence in enumerate(prevalence_per_class):
        print(f"Prevalence of class {i}: {prevalence*100:.2f}%")

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    # Example usage: Predict an image
    test_image_path = 'Test/img_4.png'  # Replace with your image path
    result = predict(test_image_path)
    if result is not None:
        predicted_digit, confidence = result
        print(f'Predicted digit: {predicted_digit}')
        print(f'Confidence: {confidence:.2f}')
        display_image_and_prediction(test_image_path, predicted_digit, confidence)
    else:
        print("Prediction failed.")
    
    # Evaluate the model on the full test dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Preprocessing test data
    x_test = x_test.astype('float32') / 255
    x_test = np.expand_dims(x_test, axis=-1)  # Reshape to (n_samples, 28, 28, 1)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # Evaluate the model on test data and display metrics
    evaluate_model_on_test_data(x_test, y_test)


# Dynamic Spatio-Temporal Self-Modeling Convolutional Gated Spiking Elastic Liquid Neural Network (DST-SM-CGSELNN)
# python deploy_and_evaluate_mnist.py
# Remarks: PASSED