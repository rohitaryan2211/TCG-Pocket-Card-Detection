import os
import cv2
import imagehash
from PIL import Image
import json  # For JSON output
import csv   # For CSV output


def calculate_image_hash(image_path):
    """Calculates the perceptual hash of an image."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image {image_path}")
            return None  # Handle unreadable images
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Ensure correct color order
        pil_image = Image.fromarray(img)
        hash_value = imagehash.phash(pil_image)
        return str(hash_value)  # Convert to string for JSON/CSV compatibility
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def process_images_in_folder(folder_path, output_file, output_format="json"):
    """
    Processes images in a folder, calculates hashes, and stores them in a file.

    Args:
        folder_path (str): Path to the folder containing images.
        output_file (str): Path to the output file (JSON or CSV).
        output_format (str): "json" or "csv" (default: "json").
    """

    image_hashes = {}  # Dictionary to store image paths and hashes
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):  # Check for image extensions
            image_path = os.path.join(folder_path, filename)
            hash_value = calculate_image_hash(image_path)
            if hash_value: # Only store if hash was successfully generated
                image_hashes[filename] = hash_value

    if output_format == "json":
        with open(output_file, "w") as f:
            json.dump(image_hashes, f, indent=4)
        print(f"Hashes saved to {output_file} in JSON format.")
    elif output_format == "csv":
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "hash"])  # Header row
            for filename, hash_value in image_hashes.items():
                writer.writerow([filename, hash_value])
        print(f"Hashes saved to {output_file} in CSV format.")
    else:
        print("Error: Invalid output format.  Must be 'json' or 'csv'.")


def main():
    folder_path = "Image_DB"  # The path to your image folder
    output_file = "image_hashes.json"  # Desired output file name
    output_format = "json"  # Choose "json" or "csv"

    process_images_in_folder(folder_path, output_file, output_format)

if __name__ == "__main__":
    main()
