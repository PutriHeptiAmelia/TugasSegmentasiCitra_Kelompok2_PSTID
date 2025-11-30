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


    #Ledi Daiyana Alfara
# FUNGSI VISUALISASI
def visualize_comparison(original, roberts, prewitt, sobel, frei_chen, title):
    plt.figure(figsize=(15, 3))
    
    plt.subplot(1, 5, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(1, 5, 2)
    plt.imshow(roberts, cmap='gray')
    plt.title('Roberts')
    plt.axis('off')
    
    plt.subplot(1, 5, 3)
    plt.imshow(prewitt, cmap='gray')
    plt.title('Prewitt')
    plt.axis('off')
    
    plt.subplot(1, 5, 4)
    plt.imshow(sobel, cmap='gray')
    plt.title('Sobel')
    plt.axis('off')
    
    plt.subplot(1, 5, 5)
    plt.imshow(frei_chen, cmap='gray')
    plt.title('Frei-Chen')
    plt.axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


# PROGRAM UTAMA
if __name__ == "__main__":
    
    file_citra_1 = 'gaussian(1).jpg'
    file_citra_2 = 'gaussian(2).jpg'
    file_citra_3 = 'SP(1).jpg'
    file_citra_4 = 'SP(2).jpg'
    
    judul_citra_1 = 'Gaussian Tingkat 1'
    judul_citra_2 = 'Gaussian Tingkat 2'
    judul_citra_3 = 'Salt & Paper Tingkat 1'
    judul_citra_4 = 'Salt & Paper Tingkat 2'
    
    # Load citra
    filenames = [file_citra_1, file_citra_2, file_citra_3, file_citra_4]
    titles = [judul_citra_1, judul_citra_2, judul_citra_3, judul_citra_4]
    images = []
    
    for filename in filenames:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"ERROR: File {filename} tidak ditemukan")
            exit()
        images.append(img)
    
    # Proses setiap citra
    for img, title in zip(images, titles):
        roberts = roberts_edge_detection(img)
        prewitt = prewitt_edge_detection(img)
        sobel = sobel_edge_detection(img)
        frei_chen = frei_chen_edge_detection(img)
        
        visualize_comparison(img, roberts, prewitt, sobel, frei_chen, title)
        
        # Simpan hasil
        clean_title = title.replace(' ', '_').replace('&', '')
        cv2.imwrite(f'{clean_title}_roberts.jpg', roberts)
        cv2.imwrite(f'{clean_title}_prewitt.jpg', prewitt)
        cv2.imwrite(f'{clean_title}_sobel.jpg', sobel)
        cv2.imwrite(f'{clean_title}_frei_chen.jpg', frei_chen)
    
    print("Selesai. Total 16 gambar hasil tersimpan.")
