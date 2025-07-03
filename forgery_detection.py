import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(img_path):
    """Loads the image, converts to grayscale, and applies histogram equalization."""
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Error: The image '{img_path}' was not found or is not a valid image.")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return image, gray, equalized

def detect_edges(image):
    """Detects edges using Sobel, Prewitt, and Laplacian filters."""
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    prewittx = cv2.filter2D(image, -1, np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]))
    prewitty = cv2.filter2D(image, -1, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]))

    combined_edges = cv2.convertScaleAbs(sobelx) + cv2.convertScaleAbs(sobely) + \
                     cv2.convertScaleAbs(laplacian) + cv2.convertScaleAbs(prewittx) + \
                     cv2.convertScaleAbs(prewitty)

    return combined_edges

def threshold_image(image):
    """Applies Otsu's thresholding for segmentation."""
    _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded

def apply_filters(image):
    """Applies Median and Adaptive Median Filters to reduce noise."""
    median_filtered = cv2.medianBlur(image, 5)
    adaptive_median = cv2.adaptiveThreshold(median_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
    return adaptive_median

def highlight_suspected_regions(original, mask):
    """Draws bounding boxes around suspected forgery regions."""
    result = original.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    suspicious_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 200:  # ignore tiny noise
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
            suspicious_count += 1
    return result, suspicious_count

def display_results(images, titles):
    """Displays images in grid format."""
    plt.figure(figsize=(15, 6))
    for i in range(len(images)):
        plt.subplot(2, 4, i+1)
        if len(images[i].shape) == 2:
            plt.imshow(images[i], cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def detect_forgery(img_path):
    """Runs the full pipeline to detect image forgery and classify it."""
    original, gray, equalized = preprocess_image(img_path)
    edges = detect_edges(equalized)
    thresholded = threshold_image(equalized)
    filtered = apply_filters(thresholded)

    # Feature Extraction
    edge_density = np.sum(edges) / edges.size
    threshold_density = np.sum(thresholded) / 255 / thresholded.size

    # Highlight suspected zones
    highlighted_image, suspected_count = highlight_suspected_regions(original, filtered)

    # Tuned decision rule
    if (edge_density > 0.2 and threshold_density > 0.5) or suspected_count > 5:
        forgery_status = "⚠ Forgery Detected"
    else:
        forgery_status = "✅ No Forgery Detected"

    print(f"Forgery Status: {forgery_status}")
    print(f"Edge Density: {edge_density:.3f}, Threshold Density: {threshold_density:.3f}, Suspected Regions: {suspected_count}")

    # Display all stages
    display_results(
        [original, gray, equalized, edges, thresholded, filtered, highlighted_image],
        ['Original', 'Grayscale', 'Histogram Equalized', 'Edges', 'Otsu Threshold', 'Filtered', 'Suspected Regions']
    )

# ✅ Run it with your uploaded image
detect_forgery('img.png')
