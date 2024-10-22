import numpy as np
import cv2
from MatchPics import matchPics
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from helper import plotMatches

# Q2.1.5
# Read the image and convert to grayscale, if necessary
I1 = cv2.imread('./data/cv_cover.jpg')
if I1.ndim == 3:
    I1_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
else:
    I1_gray = I1

# Initialize a list to store the number of matches for each rotation
match_counts = []

for i in range(36):
    print(f'{i * 10}° rotation processing...')
    # Rotate Image
    rotated_image = rotate(I1_gray, i * 10, reshape=False, mode='reflect')
    
    # Check if matchPics function is expecting color images and convert if necessary
    # This part is based on your original matchPics code that seems to expect color images
    # If your matchPics function can handle grayscale, remove the color conversion
    rotated_image_color = cv2.cvtColor(rotated_image, cv2.COLOR_GRAY2BGR)
    
    # Compute features, descriptors and match features
    matches, locs1, locs2 = matchPics(I1, rotated_image_color) # pass the original I1 if it's needed in color

    # Visualization of rotation mathcPics
    # plotMatches(I1, rotated_image_color, matches, locs1, locs2)

    # Update histogram: count the number of matches
    match_counts.append(matches.shape[0])

# Prepare data for histogram
# Flatten the match counts to match the number of bins
angles = np.repeat(range(0, 360, 10), match_counts)

# Display histogram
plt.figure()
plt.hist(angles, bins=range(0, 370, 10), edgecolor='black')  # Bins represent the rotation angles
plt.xlabel('Rotation Angle (degrees)')
plt.ylabel('Number of Matches')
plt.title('Number of Feature Matches vs Rotation Angle')
plt.xticks(range(0, 360, 10))
plt.show()


