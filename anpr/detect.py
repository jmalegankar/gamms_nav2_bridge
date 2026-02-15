import cv2
from ultralytics import YOLO
import easyocr

IMAGE_PATH = 'car.png'
LICENSE_PLATE_MODEL = 'license_plate_detector.pt'

print("Loading YOLO model...")
model = YOLO(LICENSE_PLATE_MODEL)

print("Loading OCR reader...")
reader = easyocr.Reader(['en'], gpu=True)

print(f"Reading image: {IMAGE_PATH}")
image = cv2.imread(IMAGE_PATH)

print("Detecting license plates...")
results = model(image, conf=.5)

for result in results:
    boxes = result.boxes
    
    for box in boxes:
        # Get coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        
        print(f"\nLicense Plate  detected (confidence: {confidence:.2f})")
        
        # Crop the license plate
        plate_img = image[y1:y2, x1:x2]
        
        if plate_img.size == 0:
            continue
        
        # Convert to grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding for better OCR
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        print("Reading plate text...")
        ocr_results = reader.readtext(thresh)
        
        # Extract text
        text = ''
        if ocr_results:
            text = ' '.join([r[1] for r in ocr_results])
            text = text.replace(' ', '').upper()
            print(f"Plate Text: {text}")
        else:
            print("Could not read text from this plate")
            text = "UNREADABLE"
        
        # Draw green rectangle around the LICENSE PLATE
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Put text label above the plate
        label = f"{text} ({confidence:.2f})"
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        


# Show result
cv2.imshow('ANPR Result - Press any key to close', image)
cv2.waitKey(0)
cv2.destroyAllWindows()