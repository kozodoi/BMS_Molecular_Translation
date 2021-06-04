####### SMART CROP

'''
Adapted from https://www.kaggle.com/michaelwolff/bms-inchi-cropped-img-sizes-for-best-resolution
'''

def smart_crop(img, 
               contour_min_size = 2, 
               small_stuff_size  = 2, 
               small_stuff_dist  = 5, 
               pad_pixels       = 15):
    
    # idea: pad with contour_min_size pixels just in case we cut off
    #       a small part of the structure that is separated by a missing pixel
    
    img = 255 - img
        
    _, thresh   = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    small_stuff = []
    
    x_min0, y_min0, x_max0, y_max0 = np.inf, np.inf, 0, 0
    for cnt in contours:
        if len(cnt) < contour_min_size:  # ignore contours under contour_min_size pixels
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w <= small_stuff_size and h <= small_stuff_size:  # collect position of small contours starting with contour_min_size pixels
            small_stuff.append([x, y, x+w, y+h])
            continue
        x_min0 = min(x_min0, x)
        y_min0 = min(y_min0, y)
        x_max0 = max(x_max0, x + w)
        y_max0 = max(y_max0, y + h)
        
    x_min, y_min, x_max, y_max = x_min0, y_min0, x_max0, y_max0
    
    # enlarge the found crop box if it cuts out small stuff that is very close by
    for i in range(len(small_stuff)):
        if small_stuff[i][0] < x_min0 and small_stuff[i][0] + small_stuff_dist >= x_min0:
             x_min = small_stuff[i][0]
        if small_stuff[i][1] < y_min0 and small_stuff[i][1] + small_stuff_dist >= y_min0:
             y_min = small_stuff[i][1]
        if small_stuff[i][2] > x_max0 and small_stuff[i][2] - small_stuff_dist <= x_max0:
             x_max = small_stuff[i][2]
        if small_stuff[i][3] > y_max0 and small_stuff[i][3] - small_stuff_dist <= y_max0:
             y_max = small_stuff[i][3]
                             
    if pad_pixels > 0:  # make sure we get the crop within a valid range
        y_min = max(0, y_min-pad_pixels)
        y_max = min(img.shape[0], y_max+pad_pixels)
        x_min = max(0, x_min-pad_pixels)
        x_max = min(img.shape[1], x_max+pad_pixels)
        
    img_cropped = img[y_min:y_max, x_min:x_max]
    
    img_cropped = 255 - img_cropped
    
    return img_cropped



####### PADDING

def pad_image(image):
    
    h, w = image.shape[0], image.shape[1]
    
    border_t, border_b = 0, 0
    
    if w > h:
        border_t = (w - h) // 2 
        border_b = (w - h) // 2 
        
    image = cv2.copyMakeBorder(image,
                               top        = border_t,
                               bottom     = border_b,
                               left       = 0,
                               right      = 0,
                               borderType = cv2.BORDER_CONSTANT,
                               value      = 255)
    
    return image