import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('Trained Models/ecnn_mnist.keras')

def preprocess_image(image):
    # Convert to grayscale and resize to 28x28 pixels
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=[0, -1])  # Shape to (1, 28, 28, 1)
    return image

def predict(image):
    image = preprocess_image(image)
    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction)
    return predicted_digit

def main():
    # Open the camera
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        # Display the frame
        cv2.imshow('Camera', frame)

        # Predict the digit when 'p' is pressed
        if cv2.waitKey(1) & 0xFF == ord('p'):
            digit = predict(frame)
            print(f'Predicted digit: {digit}')
        
        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


# Efficient Convolutional Neural Network (ECNN)
# python ecnn_deploy_oncam.py
# Error (always the same results)