import cv2
import numpy as np
import tensorflow as tf
import csv
#import pygame
from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog, Label, Button, Canvas, PhotoImage

# Constants
IMG_WIDTH = 30
IMG_HEIGHT = 30
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

def get_categories_from_file(labels_file):
    """
    Extract category names from a labels file (e.g., CSV or TXT).
    The file should map class indices to category names.
    """
    categories = []
    with open(labels_file, 'r') as file:
        reader = csv.reader(file)
        next(reader) 
        for row in reader:
            categories.append(row[1])  # Assuming the second column contains category names
    return categories
CATEGORIES = get_categories_from_file("labels.csv")  # Replace with actual category names if available

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
    """Main function to run tkinter GUI."""
    

def main():
    def on_select_image():
        nonlocal file_path
        file_path = select_image()
        if file_path:
            label_text.set("Image Selected")
            img = PhotoImage(file=file_path)
            canvas.image = img
            canvas.create_image(0, 0, anchor="nw", image=img)

    def on_classify():
        if file_path:
            result = predict_image(file_path)
            result_text.set(f"Recognized Sign: {result}")

    def on_reset():
        nonlocal file_path
        file_path = None
        label_text.set("Select an image to recognize a traffic sign")
        result_text.set("")
        canvas.delete("all")

    # Initialize tkinter
    root = Tk()
    root.title("Traffic Sign Recognition")

    # Variables
    file_path = None
    label_text = Label(root, text="Select an image to recognize a traffic sign")
    label_text.pack()

    # Canvas for displaying the image
    canvas = Canvas(root, width=200, height=200, bg="gray")
    canvas.pack()

    # Buttons
    select_button = Button(root, text="Select Image", command=on_select_image)
    select_button.pack()

    classify_button = Button(root, text="Classify", command=on_classify)
    classify_button.pack()

    reset_button = Button(root, text="Reset", command=on_reset)
    reset_button.pack()

    # Result label
    result_text = Label(root, text="")
    result_text.pack()

    root.mainloop()
    """Main function to run pygame GUI."""
    """pygame.init()
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
    
    pygame.quit()"""

if __name__ == "__main__":
    main()
