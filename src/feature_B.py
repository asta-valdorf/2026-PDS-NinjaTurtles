import numpy as np
import cv2

# Border function - "Is only finding and evaluating the largest spot if multiple!!!"
def border_irregularity(mask):
    '''
    * REMARK! only working properly with 1 spot, or if only the main(biggest) is wanted!!!
    - Input: Mask of a skin lesion

    - For contours:
    "RETR_EXTERNAL" only takes the outermost contour, with ignoring holes inside ()
    "CHAIN_APPROX_NONE" keeps all contour points for maximum accuracy. 

    We then find the larges countour, the border of the lesion, finds the area, finding the compactness and at last finding the irreguality score
    
    - Output: An irregularity-score from 0(perfect)-1(irregular)
    '''
    # First we find the contours(omkridser)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # np.uint8 needed for OpenCV
    # The way it works is, it finds the boundaries of the spots, where it is has value 1(lesion present) at one points, and at the one next to it has 0(not lesion present).
    # the second output "_" is for hierarchy, which is useless with RETR_ETERNAL since it only takes the outermost contour for each spot

    # If no contours found at all, return nan
    if not contours:
        return np.nan
    
    # Now we want to find the largers contour, which gives the largest spot if there is multiple (It is easier for the next parts)
    largest_contour = max(contours, key=cv2.contourArea)

    # Now we can calculate the perimeter. True is to make sure, start and endpoint, is at the same spot
    perimeter = cv2.arcLength(largest_contour, True)

    # And the area of the pixels inside the lesion
    area = np.sum(mask>0)

    # With the area, we can calculate the compactness : (Perimier^2) / (4*Pi*Area), with 1 being a perfect circle and over 1 being irregular (up to 5)
    compactness = (perimeter * perimeter) / (4 * 3.14159 * area + 1e-6) # 1e-6 is used to avoid division with 0

    # Given lesions can have higher scores than 1, up to 5, we normalize to a interval of [0,1]
    irregularity = min(max((compactness-1)/4,0), 1.0)
    
    return irregularity