import numpy as np

# Asymmetri-function - "Is taking all spots into account, so for multiple spots, the asymmetri-score will be inaccurate!!!"
def asymmetry_np_centroid(mask):
    ''' 
    -Input: Mask of a skin lesion
    
    - Creates numpy arrays, and take average to get center point (centroid) of the lesion. 
    - Then split and sum the 4 different quadrants.
    - And at last, computing the assymetry

    - Returns: Asymmetry of the mask, where 0 is perfect symmetry and 1 and completly asymmetric
    '''
    # We use centroid to find the "center of mass", to find the point where to split to 4 quadrants later
    y, x = np.where(mask > 0) # Creating 2 numpy arrays with indices where mask > 0 (lesion present)

    # Safeguard: If the mask is empty, or not loaded properly, return 0 
    if len(x) == 0 or len(y) == 0:
        return np.nan  # or return np.nan

    cx, cy = np.mean(x), np.mean(y) # "c" named after the method. Taking the average of x- and y-coordinates which gives the center point (cy = center-y-coordinate)
    
    # Spliting all mask into 4 quadrants and sum the points
    q1 = mask[:int(cy), :int(cx)].sum() # For each quadrant, slicing is used. So, if :int(cy), :int(cx), 
    q2 = mask[:int(cy), int(cx):].sum() # then everythin up to the center is included,
    q3 = mask[int(cy):, :int(cx)].sum() # and int(cy):, int(cx):, then all after the center is included.
    q4 = mask[int(cy):, int(cx):].sum() # General ":" before means before that, since it comes before int(cy)(which is the center y-coordinate), vice versa.
    
    # Asymmetry = difference between opposite quadrants
    return max(abs(q1 - q4), abs(q2 - q3)) / (q1 + q2 + q3 + q4 + 1e-6) # Formula for asymmetri 