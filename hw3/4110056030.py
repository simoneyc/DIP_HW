import cv2
import numpy as np
import random
import math

class FilterType:
    Median, AdaptiveMedian = range(2)

Source_Image_Path = "input.jpg"
window_name_1 = "Original Image"
window_name_2 = "Source Image with Impulse Noise"
window_name_3 = "Median Filter"
window_name_4 = "Adaptive Median Filter"

Pa = 1/4  # 椒鹽雜訊的黑色點概率
Pb = 1/3  # 椒鹽雜訊的白色點概率

mask_size = 7
border = (mask_size - 1) // 2
mask_element = mask_size * mask_size

def addImpulseNoise(img):
    # Impulse noise
    for i in range(border, img.shape[0] - border):
        for j in range(border, img.shape[1] - border):
            x = random.random()
            if x < Pa:
                img[i, j] = 0
            elif x < Pa + Pb:
                img[i, j] = 255

def draw_text(img, text, position, color, font_scale):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    cv2.putText(img, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

def filter(src, dst, type):
    for i in range(border, dst.shape[0] - border):
        for j in range(border, dst.shape[1] - border):
            if type == FilterType.AdaptiveMedian:
                dst[i, j] = adaptive_median_filter(src, i, j)
            else:
                mask = []
                for u in range(i - border, i + border + 1):
                    for v in range(j - border, j + border + 1):
                        mask.append(src[u, v])
                dst[i, j] = bubble_sort(mask, type)

def bubble_sort(arr, type):
    arr.sort()
    if type == FilterType.Median:
        return arr[len(arr) // 2]

def adaptive_median_filter(img, i, j, max_window_size=7):
    window_size = 3
    while window_size <= max_window_size:
        border = window_size // 2
        mask = []
        for u in range(i - border, i + border + 1):
            for v in range(j - border, j + border + 1):
                if 0 <= u < img.shape[0] and 0 <= v < img.shape[1]:
                    mask.append(img[u, v])
        mask.sort()
        Z_min = mask[0]
        Z_max = mask[-1]
        Z_med = mask[len(mask) // 2]
        if Z_min < Z_med < Z_max:
            Z_xy = img[i, j]
            if Z_min < Z_xy < Z_max:
                return Z_xy
            else:
                return Z_med
        else:
            window_size += 2
    return img[i, j]

def psnr(original, filtered):
    mse = np.mean((original - filtered) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * math.log10(max_pixel / math.sqrt(mse))

def main():
    srcImg = cv2.imread(Source_Image_Path, cv2.IMREAD_UNCHANGED)
    if srcImg is None:
        print("Error")
        return -1

    cv2.namedWindow(window_name_1, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(window_name_2, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(window_name_3, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(window_name_4, cv2.WINDOW_AUTOSIZE)

    original_image = srcImg.copy()

    # Convert to grayscale
    srcImg = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)

    random.seed()

    result_median = np.zeros_like(srcImg)
    result_adaptive_median = np.zeros_like(srcImg)
    
    noisy_image = srcImg.copy()
    addImpulseNoise(noisy_image)

    # Draw text on the noisy image
    draw_text(noisy_image, 'CSE', (10, 30), (0, 0, 0), font_scale=1.0)
    draw_text(noisy_image, 'NCHU', (600, 30), (255, 255, 255), font_scale=1.0)

    cv2.imshow(window_name_1, original_image)  # Original Image
    cv2.imshow(window_name_2, noisy_image)  # Source Image with Impulse Noise

    filter(noisy_image, result_median, FilterType.Median)
    cv2.imshow(window_name_3, result_median)  # Median Filter

    filter(noisy_image, result_adaptive_median, FilterType.AdaptiveMedian)
    cv2.imshow(window_name_4, result_adaptive_median)  # Adaptive Median Filter

    psnr_median = psnr(srcImg, result_median)
    psnr_adaptive_median = psnr(srcImg, result_adaptive_median)
    print(f"PSNR(Median Filter): {psnr_median}")
    print(f"PSNR(Adaptive Median Filter): {psnr_adaptive_median}")

    cv2.waitKey(0)
    return 0

if __name__ == "__main__":
    main()
