#!/usr/bin/env python
# coding: utf-8

# In[16]:


#to double check for run prompts


# In[2]:


import cv2
import os
import numpy as np
import pandas as pd
import pickle
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
from itertools import product
import sys

global_results = {}


# In[3]:

# Get the directory of the current script
base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))

# Load the SVM models
with open('svm_baybayin_rbf_noconfusion.sav', 'rb') as f:
    svm_model_1 = pickle.load(f)

with open('svm_baybayin_linear_confusion_v2.sav', 'rb') as f:
    svm_model_2 = pickle.load(f)

# Load the label mappings
label_map_1 = pd.read_csv('baybayin character 7.csv')
label_map_2 = pd.read_csv('baybayin character 8.csv')
id_to_label_1 = {row['id']: row['label'] for _, row in label_map_1.iterrows()}
id_to_label_2 = {row['id']: row['label'] for _, row in label_map_2.iterrows()}

# Load the CSV file with the Tagalog words
tagalog_words_df = pd.read_csv('tagalog_words.csv')
tagalog_words = tagalog_words_df['word'].values.tolist()


# In[4]:


# Add a function to find the closest word
def closest_word(input_word):
    similarity = [SequenceMatcher(None, input_word, word).ratio() for word in tagalog_words]
    index = np.argmax(similarity)
    return tagalog_words[index]


# In[5]:


def extract_segments(img_gray):
    # Preprocess the image for character segmentation
    img_binary = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, img_binary = cv2.threshold(img_binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_binary = cv2.Canny(img_binary, 100, 200)

    # Find contours
    contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours from left to right
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    contours, bounding_boxes = zip(*sorted(zip(contours, bounding_boxes), key=lambda b: b[1][0]))
    
    # Extract segments from the grayscale image
    segments = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        segment = img_gray[y:y+h, x:x+w]
        segments.append(segment)
        
    return segments


# In[6]:


def separate_characters(predicted_chars):
    separated_chars = []
    for char in predicted_chars:
        if char == 'dara':
            separated_chars.append(['da', 'ra'])
        elif char == 'ei':
            separated_chars.append(['e', 'i'])
        elif char == 'ou':
            separated_chars.append(['o', 'u'])
        else:
            separated_chars.append([char])
    return separated_chars


# In[7]:


def recombine_characters(separated_chars):
    combinations = list(product(*separated_chars))
    words = [''.join(combo) for combo in combinations]
    closest_words = [closest_word(word) for word in words]
    similarity_scores = [SequenceMatcher(None, word, closest_word).ratio() for word, closest_word in zip(words, closest_words)]
    best_word_index = np.argmax(similarity_scores)
    return closest_words[best_word_index]


# In[8]:


def detect_dots(img_gray):
    # Preprocess the image for contour detection
    img_binary = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, img_binary = cv2.threshold(img_binary, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours from left to right
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    contours, bounding_boxes = zip(*sorted(zip(contours, bounding_boxes), key=lambda b: b[1][0]))

    # Initialize lists to store the characters and the dots
    characters = []
    dots = []

    # Create a mask to remove dots
    dot_mask = np.ones(img_gray.shape, dtype=np.uint8)*255

    # Loop over the contours and categorize them
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Adjusted threshold to detect slightly bigger dots
        if w*h > 100:
            characters.append({'contour': contour, 'x': x, 'y': y, 'w': w, 'h': h, 'dot': 'no_dot'})
        else:
            dots.append({'contour': contour, 'x': x, 'y': y, 'w': w, 'h': h})
            cv2.drawContours(dot_mask, [contour], -1, 0, -1)  # remove dot from the mask
    
    # Loop over the characters and check for dots above or below them
    for char in characters:
        char_mid = char['x'] + char['w'] / 2
        for dot in dots:
            dot_mid = dot['x'] + dot['w'] / 2
            # Relaxed criteria for matching dots to characters
            if abs(char_mid - dot_mid) < char['w'] * 0.5:  # check within half of character width
                if dot['y'] < char['y']:
                    char['dot'] = 'above'
                elif dot['y'] > (char['y'] + char['h']):  # check if dot is below character
                    char['dot'] = 'below'
    
    # Apply the mask to the image
    img_gray_no_dots = cv2.bitwise_and(img_gray, img_gray, mask=dot_mask)

    # Extract the image segments and dot positions
    segments = [img_gray_no_dots[char['y']:char['y']+char['h'], char['x']:char['x']+char['w']] for char in characters]
    dot_positions = [char['dot'] for char in characters]

    return segments, dot_positions


# In[9]:


def detect_plus_signs(img_gray):
    # Preprocess the image for contour detection
    img_binary = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, img_binary = cv2.threshold(img_binary, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours from left to right
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    contours, bounding_boxes = zip(*sorted(zip(contours, bounding_boxes), key=lambda b: b[1][0]))

    characters = []
    plus_signs = []
    matched_plus_indices = []

    plus_mask = np.ones(img_gray.shape, dtype=np.uint8) * 255

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        aspect_ratio = float(w) / h
        # Check the aspect ratio and the area threshold
        if (0.5 <= aspect_ratio <= 1.5) and (100 <= w * h <= 400):
            plus_signs.append({'contour': contour, 'x': x, 'y': y, 'w': w, 'h': h})
            cv2.drawContours(plus_mask, [contour], -1, 0, -1)  # remove plus sign from the mask
        else:
            characters.append({'contour': contour, 'x': x, 'y': y, 'w': w, 'h': h, 'plus': 'no_plus'})
    
    for char in characters:
        char_mid = char['x'] + char['w'] / 2
        for i, plus in enumerate(plus_signs):
            if i in matched_plus_indices:
                continue
            
            plus_mid = plus['x'] + plus['w'] / 2
            if abs(char_mid - plus_mid) < char['w'] * 0.7:
                if plus['y'] > (char['y'] + char['h']):
                    char['plus'] = 'below'
                    matched_plus_indices.append(i)
    
    img_gray_no_plus = cv2.bitwise_and(img_gray, img_gray, mask=plus_mask)
    segments = [img_gray_no_plus[char['y']:char['y'] + char['h'], char['x']:char['x'] + char['w']] for char in characters]
    plus_positions = [char['plus'] for char in characters]

    return segments, plus_positions


# In[10]:


def modify_character_with_plus_and_dots(characters, dots, pluses):
    modified_characters = []  # Initialize an empty list for modified characters

    for i, (segment_predictions, dot, plus) in enumerate(zip(characters, dots, pluses)):
        print(f"Processing segment {i} with dot: {dot}, plus: {plus}")  # Add this logging
        segment_modified = []
        for char in segment_predictions:
            if char.endswith('a'):  # Check if character end s with 'a'
                if dot == 'above':
                    segment_modified.extend([char[:-1] + 'e', char[:-1] + 'i'])  # Replace 'a' with 'e' and 'i'
                elif dot == 'below' and plus != 'below':  # Consider the dot only if there's no confirmed plus
                    segment_modified.extend([char[:-1] + 'o', char[:-1] + 'u'])  # Replace 'a' with 'o' and 'u'
                elif plus == 'below':
                    segment_modified.append(char[:-1])  # Remove the 'a' from the character
                else:
                    segment_modified.append(char)
            else:
                segment_modified.append(char)

        modified_characters.append(segment_modified)

    return modified_characters


# In[11]:


def recombine_modified_characters(modified_characters):
    # Flatten each set of predictions for each segment into a single list
    flat_predictions = [prediction for segment in modified_characters for prediction in segment]
    
    # Generate all combinations of these predictions
    combinations = list(product(*modified_characters))
    
    # Join the characters of each combination to form possible words
    possible_words = [''.join(combo) for combo in combinations]
    
    return possible_words


# In[12]:


def plus_or_dot(segment):
    # Convert segment to binary
    _, binary_segment = cv2.threshold(segment, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(binary_segment, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=0, maxRadius=0)

    # If circles are detected
    if circles is not None:
        # Convert the (x, y) coordinates and radius of the circles to integers
        circles = np.uint16(np.around(circles))
        
        # Assuming there's only one significant circle, but you can loop through and analyze more if needed
        circle = circles[0, 0]
        _, _, radius = circle
        
        # Define a range for dot's expected radius. This may need tweaking.
        min_dot_radius = 2   # Example value; adjust as needed
        max_dot_radius = 15  # Example value; adjust as needed
        
        if min_dot_radius <= radius <= max_dot_radius:
            return True  # It's a dot
    return False  # It's a plus sign


# In[13]:


def predict_character(img_path):
    # Load the test image and convert to grayscale
    test_img = cv2.imread(img_path)
    test_img_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    # Initialize the lists to store the predictions
    predictions = []
    raw_predictions = []
    final_predictions = []

    # Extract segments and detect dots and plus signs
    segments, dots = detect_dots(test_img_gray)
    _, pluses = detect_plus_signs(test_img_gray)

    print(f'Segments: {len(segments)}')
    print(f'Dots: {dots}')
    print(f'Pluses: {pluses}')

    # Process each segment and perform prediction
    for i, (segment, dot_position, plus_position) in enumerate(zip(segments, dots, pluses)):

        probabilities_1 = probabilities_2 = probabilities_3 = None  # Initialize probabilities with None
        # Preprocess the segment
        segment = cv2.GaussianBlur(segment, (5, 5), 0)
        _, segment = cv2.threshold(segment, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        segment_resized = cv2.resize(segment, (50, 50))
        segment_flattened = np.array(segment_resized).flatten().reshape(1, -1)

        # Perform prediction on the flattened image using all models
        predicted_labels_1 = svm_model_1.predict(segment_flattened)
        predicted_labels_2 = svm_model_2.predict(segment_flattened)

        # Get the probabilities of each prediction from all models
        probabilities_1 = svm_model_1.predict_proba(segment_flattened)
        probabilities_2 = svm_model_2.predict_proba(segment_flattened)

        # Implement voting system based on weighted average of probabilities
        pred_1, pred_2 = predicted_labels_1[0], predicted_labels_2[0]
        prob_1, prob_2 = probabilities_1[0], probabilities_2[0]

        # Combine probabilities using a weighted average
        padded_prob_2 = np.pad(prob_2, [(0, 7)], mode='constant', constant_values=0)
        weighted_probabilities = (prob_1 + padded_prob_2) / 2

        # Take the prediction with the highest weighted probability
        predicted_label = np.argmax(weighted_probabilities)
        
        # Handle the confused pairs
        if id_to_label_1[pred_1] == 'a' and id_to_label_2[pred_2] == 'ma':
            final_predictions.append('a' if prob_1[1] > prob_2[1] else 'ma')
        elif id_to_label_1[pred_1] == 'ma' and id_to_label_2[pred_2] == 'a':
            final_predictions.append('ma' if prob_1[1] > prob_2[1] else 'a')
        elif id_to_label_1[pred_1] == 'pa' and id_to_label_2[pred_2] == 'ya':
            final_predictions.append('pa' if prob_1[1] > prob_2[1] else 'ya')
        elif id_to_label_1[pred_1] == 'ya' and id_to_label_2[pred_2] == 'pa':
            final_predictions.append('ya' if prob_1[1] > prob_2[1] else 'pa')
        elif id_to_label_1[pred_1] == 'la' and id_to_label_2[pred_2] == 'ta':
            final_predictions.append('la' if prob_1[1] > prob_2[1] else 'ta')
        elif id_to_label_1[pred_1] == 'ta' and id_to_label_2[pred_2] == 'la':
            final_predictions.append('ta' if prob_1[1] > prob_2[1] else 'la')
        elif id_to_label_1[pred_1] in ['e', 'i'] and id_to_label_2[pred_2] == 'ka':
            final_predictions.append('e' if prob_1[1] > prob_2[1] else 'ka')
        elif id_to_label_1[pred_1] == 'ka' and id_to_label_2[pred_2] in ['e', 'i']:
            final_predictions.append('ka' if prob_1[1] > prob_2[1] else 'e')
        elif id_to_label_1[pred_1] == 'ha' and id_to_label_2[pred_2] == 'sa':
            final_predictions.append('ha' if prob_1[1] > prob_2[1] else 'sa')
        elif id_to_label_1[pred_1] == 'sa' and id_to_label_2[pred_2] == 'ha':
            final_predictions.append('sa' if prob_1[1] > prob_2[1] else 'ha')
        else:
            max_prob = max(prob_1.max(), prob_2.max())
            if max_prob == prob_1.max():
                final_predictions.append(id_to_label_1[pred_1])
                predictions.append(id_to_label_1[pred_1])
                raw_predictions.append(id_to_label_1[pred_1])
            elif max_prob == prob_2.max():
                final_predictions.append(id_to_label_2[pred_2])
                predictions.append(id_to_label_2[pred_2])
                raw_predictions.append(id_to_label_2[pred_2])
        
        # Check for dots
        if dot_position == 'above':
            raw_predictions.insert(i+1, 'dot')
        elif dot_position == 'below':
            raw_predictions.append('dot')

        # Check for plus signs
        if plus_position == 'below':
            raw_predictions.append('+')

        # Check for conflicting dots and pluses
        if dot_position == 'below' and plus_position == 'below':
            is_dot = plus_or_dot(segment)
            if is_dot:
                print(f"Segment {i}: Detected and confirmed dot below.")
                # Modify the character prediction for the dot
                raw_predictions[-2] = 'dot'
                raw_predictions.pop(-1)  # Remove the '+'
            else:
                print(f"Segment {i}: Detected and confirmed plus sign.")
                # Confirm the plus sign
                raw_predictions[-1] = '+'
                raw_predictions.pop(-2)  # Remove the 'dot'
                
   # Compile the predictions into a word
    predicted_word = ''.join(predictions)

    # Separate characters like 'dara', 'ei', and 'ou'
    separated_predictions = separate_characters(final_predictions)

    # Modify characters based on dot and plus positions
    modified_characters = modify_character_with_plus_and_dots(separated_predictions, dots, pluses)

    # Recombine the modified characters
    recombined_modified_characters = recombine_modified_characters(modified_characters)

    # Find the closest match for each word in the list of possible words
    closest_matching_words = [closest_word(word) for word in recombined_modified_characters]

    # Among the closest matches, select the word with the highest similarity
    similarity_scores = [SequenceMatcher(None, word, closest_matching_word).ratio() for word, closest_matching_word in zip(recombined_modified_characters, closest_matching_words)]
    best_match_index = np.argmax(similarity_scores)
    best_prediction = recombined_modified_characters[best_match_index]
    
    global_results['predicted_word'] = best_prediction
    global_results['all_possible_words'] = recombined_modified_characters
    # If recombined_modified_characters is a list, then you can convert it to a comma-separated string:
    global_results['all_possible_words_str'] = ', '.join(recombined_modified_characters)

    return best_prediction, recombined_modified_characters, test_img, raw_predictions


# In[ ]:




