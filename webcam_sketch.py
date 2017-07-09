"""
Webcam Sketch

Having fun generating a line drawing of the live webcam.  Requires a webcam (obviously), and OpenCV.  Python 3
"""
import cv2
import numpy as np

# Our sketch generating function
def sketch(image):
    # Convert image to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Clean up image using Guassian Blur
    img_gray_blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    
    # Extract edges
    canny_edges = cv2.Canny(img_gray_blur, 10, 90)
    
    kernel = np.ones((3,3), np.uint8)
    dilation = cv2.dilate(canny_edges, kernel, iterations = 1)
    
    # Do an invert binarize the image 
    mask = cv2.bitwise_not(dilation)
    mask = cv2.bitwise_or(mask, img_gray) # adds an interesting gray effect
    return mask


# Initialize webcam, cap is the object provided by VideoCapture
# It contains a boolean indicating if it was sucessful (ret)
# It also contains the images collected from the webcam (frame)
cap = cv2.VideoCapture(0)

print('Press Enter to exit')
while True:
    try:
        ret, frame = cap.read()
        cv2.imshow('Our Live Sketcher', sketch(frame))
        if cv2.waitKey(1) == 13: #13 is the Enter Key
            break
    except:
        # Release camera and close windows
        cap.release()
        cv2.destroyAllWindows()

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()