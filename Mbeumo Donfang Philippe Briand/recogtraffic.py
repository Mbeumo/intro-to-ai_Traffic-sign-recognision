import cv2
import numpy as np
import tensorflow as tf
import pygame
from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog

# Constants
IMG_WIDTH = 30
IMG_HEIGHT = 30
CATEGORIES = [str(i) for i in range(43)]  # Replace with actual category names if available
MODEL_PATH = "model.h5"  # Update with your model path

# Load the trained model
model = load_model(MODEL_PATH)

def preprocess_image(image_path):
    """Load and preprocess the image."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = image / 255.0  # Normalize
    return np.expand_dims(image, axis=0)  # Add batch dimension

def predict_image(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    print("Probabilities:", prediction)  # Debugging
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]  # Get confidence score

    print("Predicted Class Index:", predicted_class, "Confidence:", confidence)  # Debugging
    return CATEGORIES[predicted_class]

def select_image():
    """Open file dialog to select an image."""
    Tk().withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename()
    return file_path

def main():
    """Main function to run pygame GUI."""
    pygame.init()
    screen = pygame.display.set_mode((800, 400))
    pygame.display.set_caption("Traffic Sign Recognition")

    font = pygame.font.Font(None, 36)
    label_text = "Select an image to recognize a traffic sign"
    image_surface = None
    result_text = ""
    file_path = None
    
    # Buttons
    classify_button = pygame.Rect(550, 100, 200, 50)
    reset_button = pygame.Rect(550, 200, 200, 50)
    select_button = pygame.Rect(550, 300, 200, 50)

    running = True
    while running:
        screen.fill((200, 200, 200))
        text_surface = font.render(label_text, True, (0, 0, 0))
        screen.blit(text_surface, (50, 50))

        if image_surface:
            screen.blit(image_surface, (50, 100))
        
        # Draw buttons
        pygame.draw.rect(screen, (0, 150, 0), classify_button)
        pygame.draw.rect(screen, (200, 0, 0), reset_button)
        pygame.draw.rect(screen, (0, 0, 200), select_button)
        
        classify_text = font.render("Classify", True, (255, 255, 255))
        reset_text = font.render("Reset", True, (255, 255, 255))
        select_text = font.render("Select Image", True, (255, 255, 255))
        
        screen.blit(classify_text, (classify_button.x + 50, classify_button.y + 10))
        screen.blit(reset_text, (reset_button.x + 70, reset_button.y + 10))
        screen.blit(select_text, (select_button.x + 40, select_button.y + 10))
        
        # Display result
        result_surface = font.render(result_text, True, (0, 0, 0))
        screen.blit(result_surface, (50, 350))
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if select_button.collidepoint(event.pos):
                    file_path = select_image()
                    if file_path:
                        label_text = "Image Selected"
                        image = pygame.image.load(file_path)
                        image = pygame.transform.scale(image, (200, 200))
                        image_surface = image
                elif classify_button.collidepoint(event.pos) and file_path:
                    label_text = "Recognizing..."
                    result_text = f"Recognized Sign: {predict_image(file_path)}"
                elif reset_button.collidepoint(event.pos):
                    file_path = None
                    image_surface = None
                    label_text = "Select an image to recognize a traffic sign"
                    result_text = ""
        
        pygame.display.flip()
    
    pygame.quit()

if __name__ == "__main__":
    main()
