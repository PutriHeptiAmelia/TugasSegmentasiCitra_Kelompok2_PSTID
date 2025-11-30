# TugasSegmentasiCitra_Kelompok2_PSTID

#Putri Hepti Amelia
# METODE 1: OPERATOR ROBERTS
def roberts_edge_detection(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    img = image.astype(np.float64)
    rows, cols = img.shape
    result = np.zeros((rows-1, cols-1), dtype=np.float64)
    
    for y in range(rows-1):
        for x in range(cols-1):
            z1 = img[y, x]
            z2 = img[y, x+1]
            z3 = img[y+1, x]
            z4 = img[y+1, x+1]
            result[y, x] = np.sqrt((z1 - z4)**2 + (z2 - z3)**2)
    
    result = np.uint8(result / result.max() * 255)
    return result


# METODE 2: OPERATOR PREWITT
def prewitt_edge_detection(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    img = image.astype(np.float64)
    rows, cols = img.shape
    result = np.zeros((rows-2, cols-2), dtype=np.float64)
    
    for y in range(1, rows-1):
        for x in range(1, cols-1):
            gx = (img[y-1, x-1] + img[y, x-1] + img[y+1, x-1]) - \
                 (img[y-1, x+1] + img[y, x+1] + img[y+1, x+1])
            gy = (img[y+1, x-1] + img[y+1, x] + img[y+1, x+1]) - \
                 (img[y-1, x-1] + img[y-1, x] + img[y-1, x+1])
            result[y-1, x-1] = np.sqrt(gx**2 + gy**2)
    
    result = np.uint8(result / result.max() * 255)
    return result

    # Valerie Alana Yusri
    # METODE 3: OPERATOR SOBEL
def sobel_edge_detection(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    img = image.astype(np.float64)
    rows, cols = img.shape
    result = np.zeros((rows-2, cols-2), dtype=np.float64)
    
    for y in range(1, rows-1):
        for x in range(1, cols-1):
            gx = (img[y-1, x+1] + 2*img[y, x+1] + img[y+1, x+1]) - \
                 (img[y-1, x-1] + 2*img[y, x-1] + img[y+1, x-1])
            gy = (img[y-1, x-1] + 2*img[y-1, x] + img[y-1, x+1]) - \
                 (img[y+1, x-1] + 2*img[y+1, x] + img[y+1, x+1])
            result[y-1, x-1] = np.sqrt(gx**2 + gy**2)
    
    result = np.uint8(result / result.max() * 255)
    return result


# METODE 4: OPERATOR FREI-CHEN
def frei_chen_edge_detection(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    img = image.astype(np.float64)
    rows, cols = img.shape
    result = np.zeros((rows-2, cols-2), dtype=np.float64)
    sqrt2 = np.sqrt(2)
    
    for y in range(1, rows-1):
        for x in range(1, cols-1):
            gx = (img[y-1, x+1] + sqrt2*img[y, x+1] + img[y+1, x+1]) - \
                 (img[y-1, x-1] + sqrt2*img[y, x-1] + img[y+1, x-1])
            gy = (img[y-1, x-1] + sqrt2*img[y-1, x] + img[y-1, x+1]) - \
                 (img[y+1, x-1] + sqrt2*img[y+1, x] + img[y+1, x+1])
            result[y-1, x-1] = np.sqrt(gx**2 + gy**2)
    
    result = np.uint8(result / result.max() * 255)
    return result

