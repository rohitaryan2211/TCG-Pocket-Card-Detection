# Pokémon TCG Pocket Card Detection

## Overview

This repository contains a Python-based project designed to detect and identify Pokémon cards from screenshots of the Pokémon TCG Pocket mobile game. It's a fun, personal project that utilizes computer vision techniques to automatically recognize cards displayed in the game.

**Please note:** This project is intended for educational and personal use only.

## Project Structure

The project consists of the following key components:

*   `card_detection.py`: The main script responsible for detecting card contours, performing perspective correction, and identifying cards using perceptual hashing.  It loads card hash data from a JSON file.
*   `hash_generator.py`: (Not Shared - See Disclaimer Below) This script (which I am not able to provide) would be used to generate perceptual hashes for images of individual cards and save them to a JSON file, which can be used in `card_detection.py`
*   `image_hashes.json`: This file stores a dictionary of card names and their corresponding perceptual hashes. This file would be the output from `hash_generator.py`
*   `README.md`: This file, providing an overview of the project and instructions for use.

## Disclaimer Regarding Card Images and Legal Considerations

Due to copyright and legal concerns, I am **unable to provide the `Images_DB` folder**. This folder would contain images of all the Pokémon cards available in TCG Pocket, which are necessary to generate the perceptual hashes for card identification.

Without this folder, you will not be able to run `hash.py` (or the provided `hash_generator.py`) to create the `image_hashes.json` file that is used by `card_detection.py`. You will need to create the `ImagesDB` folder and collect the card images yourself if you want to use all functions of this program.

## Core Functionality

1.  **Card Detection:** The `card_detection.py` script uses OpenCV to:
    *   Preprocess the input image (grayscale conversion, edge detection).
    *   Detect contours that resemble card shapes.
    *   Apply perspective transform to warp the detected cards into a standard view.

2.  **Card Identification:**
    *   Calculates the perceptual hash of each warped card image using the `imagehash` library.
    *   Compares the calculated hash against a database of known card hashes (loaded from `image_hashes.json`).
    *   Identifies the card based
