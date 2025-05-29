# Erosi-Dilasi
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load gambar tulisan hitam putih (biner)
img = cv2.imread('karakter.png', cv2.IMREAD_GRAYSCALE)
_, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

# Definisi 9 structuring elements (3x3)
strel_list = {
    "Kotak": np.ones((3,3), dtype=np.uint8),
    "Salib": np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8),
    "Diagonal \\": np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=np.uint8),
    "Diagonal /": np.array([[0,0,1],[0,1,0],[1,0,0]], dtype=np.uint8),
    "Kotak tanpa sudut": np.array([[0,1,1],[1,1,1],[1,1,0]], dtype=np.uint8),
    "Sudut kanan atas": np.array([[0,0,1],[0,1,1],[1,1,1]], dtype=np.uint8),
    "Sudut kanan bawah": np.array([[1,1,1],[0,1,1],[0,0,1]], dtype=np.uint8),
    "Sudut kiri atas": np.array([[1,0,0],[1,1,0],[1,1,1]], dtype=np.uint8),
    "Sudut kiri bawah": np.array([[1,1,1],[1,1,0],[1,0,0]], dtype=np.uint8)
}

# Fungsi plot hasil
def plot_images(title, images, cols=3):
    plt.figure(figsize=(15,10))
    for i, (name, img) in enumerate(images.items()):
        plt.subplot(len(images)//cols+1, cols, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(name)
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

# Erosi dan Dilasi dengan setiap strel
hasil_erosi = {}
hasil_dilasi = {}

for name, strel in strel_list.items():
    erosi = cv2.erode(img_bin, strel, iterations=1)
    dilasi = cv2.dilate(img_bin, strel, iterations=1)
    hasil_erosi[name] = erosi
    hasil_dilasi[name] = dilasi

# Plot hasil
plot_images("Hasil Erosi dengan 9 Strel", hasil_erosi)
plot_images("Hasil Dilasi dengan 9 Strel", hasil_dilasi)
