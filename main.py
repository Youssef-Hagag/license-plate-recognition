from easyocr import Reader
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import arabic_reshaper
from bidi.algorithm import get_display

# Function to render Arabic text on the image
def put_arabic_text(image, text, position, color=(0, 255, 0), thickness=1):
    font_file = "Sahel.ttf"  # File containing the Arabic font
    font = ImageFont.truetype(font_file, 18)  # Load the font with size 18
    reshaped_text = arabic_reshaper.reshape(text)  # Reshape Arabic text for correct display
    bidi_text = get_display(reshaped_text)  # Get bidi text for correct RTL display
    
    # Convert the OpenCV image (numpy array) to a Pillow Image
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Use Pillow to draw the Arabic text on the image
    draw = ImageDraw.Draw(pil_image)
    draw.text(position, bidi_text, color, font=font, stroke_width=thickness)  # Draw the text
    
    # Convert the modified Pillow image back to a numpy array
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return image

# Preprocessing steps for the image before plate detection
def pre_process_image(image_path):
    # Read the image from the specified path
    car = cv2.imread(image_path)
    
    # Resize the image to a specific size (600x400 pixels)
    resized_image = cv2.resize(car, (600, 400))
    
    # Convert the resized image to grayscale
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter for noise removal while keeping the edges sharp
    noise_removal = cv2.bilateralFilter(gray, 11, 17, 17)
    
    return noise_removal

# Detect license plate using contours
def plate_detection_using_contours():
    car = pre_process_image('cars/car24.jpg')
    edged = cv2.Canny(car, 10, 200)
    
    # Find contours in the edged image
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort the contours by area in descending order and select the top 5 contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # Iterate through the detected contours
    for contour in contours:
        # Calculate the arc length of the contour
        arc_length = cv2.arcLength(contour, True)
        
        # Approximate the contour shape to a polygonal curve
        approx = cv2.approxPolyDP(contour, 0.02 * arc_length, True)
        
        # Check if the approximated polygon has 4 sides (likely a license plate)
        if len(approx) == 4:
            plate_contour = approx
            break  # Stop iterating if a valid plate contour is found
    
    # Get the bounding rectangle of the plate contour
    (x, y, w, h) = cv2.boundingRect(plate_contour)
    
    # Extract the region of interest (ROI) containing the detected plate
    plate = car[y:y + h, x:x + w]
    return plate

# Detect license plate using Haar Cascade Classifier
def plate_detection_haarcascade():
    # Preprocess the image before plate detection
    car = pre_process_image('cars/car24.jpg')
    
    # Load the pre-trained Haar Cascade classifier for license plates
    plate_cascade = cv2.CascadeClassifier("haarcascade_license_plate.xml")
    
    # Detect potential plates in the preprocessed image
    plates = plate_cascade.detectMultiScale(car, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around detected license plates
    for (x, y, w, h) in plates:
        cv2.rectangle(car, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Extract the region of the first detected plate
    for (x, y, w, h) in plates:
        plate_region = car[y:y + h, x:x + w]
        break  # Extract the first detected plate only
    
    return plate_region

# Perform plate detection using both methods
plate1 = plate_detection_using_contours()
plate2 = plate_detection_haarcascade()

# Perform OCR to read text from the detected plates
reader = Reader(['ar'], gpu=False, verbose=False)
detection1 = reader.readtext(plate1)
detection2 = reader.readtext(plate2)
print(detection1)
print(detection2)

# Display the detected plates
cv2.imshow('License Plate 1', plate1)
cv2.imshow('License Plate 2', plate2)
cv2.waitKey(0)

# if len(detection) == 0:
#     text = "Impossible to read the text from the license plate"
#     cv2.putText(car, text, (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 0, 255), 3)
#     cv2.drawContours(car, [plate_cnt], -1, (0, 255, 0), 3)
#     cv2.putText(car, '', (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0), 2)
#     cv2.imshow('Image', car)
#     cv2.waitKey(0)
# else:
#     cv2.drawContours(car, [plate_cnt], -1, (0, 255, 0), 3)
#     text = f"{detection[0][1]} {detection[0][2] * 100:.2f}%"
#     print(text)
#     car = put_arabic_text(car, text, (x, y - 30), (0, 255, 0), 1)
#     cv2.imshow('license plate', plate)
#     cv2.imshow('Image', car)
#     cv2.waitKey(0)

