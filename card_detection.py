import cv2
import numpy as np
import imagehash
from PIL import Image
import matplotlib.pyplot as plt
import json
import os

def preprocess_image(image_path):
    """Loads and preprocesses the image (grayscale conversion)."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray


def detect_card_contours(gray_image):
    """Detects contours that resemble playing cards."""
    edges = cv2.Canny(gray_image, 2, 50)  # Adjust thresholds as needed
    # plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
    # plt.title("Edges (Outlined)")
    # plt.show()
    kernel = np.ones((2, 2), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=2)
    # plt.imshow(cv2.cvtColor(dilated_edges, cv2.COLOR_BGR2RGB))
    # plt.title("Edges")
    # plt.show()
    contours, _ = cv2.findContours(dilated_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    card_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        # print(contour)
        # print(area)
        if 1000 < area < 50000:  # Adjust area thresholds as needed
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) == 4:
                card_contours.append(approx)
    
    #Orignal card contours had trouble not able to detect left top corner for each card
    #This is a fix for the card contours
    
    card_rectangles = [] 

    for card_contour in card_contours:
        minx = min(card_contour[0][0][0], card_contour[1][0][0], card_contour[2][0][0], card_contour[3][0][0])
        maxx = max(card_contour[0][0][0], card_contour[1][0][0], card_contour[2][0][0], card_contour[3][0][0])
        miny = min(card_contour[0][0][1], card_contour[1][0][1], card_contour[2][0][1], card_contour[3][0][1])
        maxy = max(card_contour[0][0][1], card_contour[1][0][1], card_contour[2][0][1], card_contour[3][0][1])

        card_rectangle = np.array([[[minx, miny]], [[maxx, miny]], [[maxx, maxy]], [[minx, maxy]]], dtype="int32")
        card_rectangles.append(card_rectangle)

    return card_rectangles
    # return card_contours

def display_outlined_cards(image, card_contours):
    """Displays the original image with the detected card contours outlined."""
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, card_contours, -1, (0, 0, 255), 5)  # Red contours
    plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
    plt.title("Cards Detected (Outlined)")
    plt.show()

def warp_card(image, card_contour):
    """Applies perspective transform to warp the card."""
    pts = card_contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left

    (tl, tr, br, bl) = rect
    # print(tl, tr, br, bl)
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def calculate_hash(image):
    """Calculates the perceptual hash of an image."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    hash = imagehash.phash(pil_image)
    return hash

def identify_card(card_hash, card_database, threshold=20):
    """Identifies the card based on its hash and a database of known hashes."""
    closest_match = None
    min_distance = threshold

    for card_name, db_hash_str in card_database.items():
        try:
            db_hash = imagehash.hex_to_hash(db_hash_str)
            distance = card_hash - db_hash  # Hamming distance
            if distance < min_distance:
                min_distance = distance
                closest_match = card_name
        except ValueError as e:
            print(f"Error converting database hash for {card_name}: {e}")
            continue  # Skip to the next card in the database

    return closest_match, min_distance

def load_card_database(json_file):
    """Loads the card database from a JSON file."""
    try:
        with open(json_file, "r") as f:
            card_database = json.load(f)
        return card_database
    except FileNotFoundError:
        print(f"Error: Could not find card database file: {json_file}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in card database file: {json_file}")
        return {}

def detect_cards(image_path, card_database):

    image, gray = preprocess_image(image_path)
    if image is None or gray is None:
        return []

    card_contours = detect_card_contours(gray)

    # Display the outlined cards before warping
    display_outlined_cards(image, card_contours)

    warped_cards = []
    for card_contour in card_contours:
        warped_card = warp_card(image, card_contour)
        warped_cards.append(warped_card)

    results = []
    for warped_card in warped_cards:
        card_hash = calculate_hash(warped_card)
        match, distance = identify_card(card_hash, card_database)
        results.append((warped_card, match, distance))

    return results


def main():
    # Load Card Database from JSON
    json_file = "image_hashes.json"  # Replace with your JSON file path
    card_database = load_card_database(json_file)

    if not card_database:
        print("Card database is empty. Please generate the database first.")
        return
    
    for i in range(1, 8):
        image_path = f"Screenshots/S{i}.png"
        detected_cards = detect_cards(image_path, card_database)

        # Display the results
        if detected_cards:
            for i, (card_image, card_name, confidence) in enumerate(detected_cards):
                plt.figure()
                plt.imshow(cv2.cvtColor(card_image, cv2.COLOR_BGR2RGB))
                title = f"Card {i+1}: "
                if card_name:
                    title += f"{card_name} (Confidence: {confidence})"
                else:
                    title += "Unknown"
                plt.title(title)
                plt.show()
        else:
            print("No cards detected.")


if __name__ == "__main__":
    main()
