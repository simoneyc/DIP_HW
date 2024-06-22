import cv2
import numpy as np
import math

# 定義卷積核
masks = np.array([
    [1, 1, 1,
     1, 1, 1,
     1, 1, 1],

    [-1, -2, -1,
      0,  0,  0,
      1,  2,  1],

    [-1, 0, 1,
     -2, 0, 2,
     -1, 0, 1],

    # [0, -1, 0,
    #  -1,  4, -1,
    #  0, -1, 0]
    #  ,

    [-1, -1, -1,
     -1,  8, -1,
     -1, -1, -1]
]).reshape(4, 3, 3)

source_image_path = "A.jpg"

def convolution(src, mask):
    dst = np.zeros_like(src)
    for i in range(1, src.shape[0] - 1):
        for j in range(1, src.shape[1] - 1):
            region = src[i-1:i+2, j-1:j+2]
            convo = np.sum(region * mask)
            dst[i, j] = np.clip(convo, 0, 255)
    return dst

def calculate_psnr(original, compared):
    mse = np.mean((original - compared) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(255.0 / math.sqrt(mse))
    return psnr

# 讀取影像
src_img = cv2.imread(source_image_path)
src_temp = src_img

if src_img is None:
    print("Error! Can not open the Image file!")
else:
    cv2.imshow("Source Image", src_img)

# cv2.namedWindow("First-order differential", cv2.WINDOW_A


# 轉換成灰階影像
src_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray level", src_gray)

# 初始化結果矩陣
diff_1 = np.zeros_like(src_gray)
diff_2 = np.zeros_like(src_gray)
mean_f = np.zeros_like(src_gray)

# 進行卷積操作
diff_1 = convolution(src_gray, masks[1]) # Sobel y
diff_2 = convolution(src_gray, masks[2]) # Sobel x

# 計算Sobel邊緣檢測結果
sobel = np.clip(diff_1 + diff_2, 0, 255)
diff_1 = sobel

# 計算Laplacian邊緣檢測結果
laplacian = convolution(src_gray, masks[3]) # Laplacian
diff_2 = laplacian
mean_f = convolution(diff_1, masks[0])  # Mean filter on Sobel result -> Unsharp mask

# Unsharp masking
for i in range(1, diff_2.shape[0] - 1):
    for j in range(1, diff_2.shape[1] - 1):
        src_img[i, j] = np.clip(src_img[i, j] + diff_2[i, j] * (mean_f[i, j] / 255), 0, 255)


# 灰度圖像與Sobel結果相加
gray_plus_sobel = cv2.add(src_gray, diff_1)

# 灰度圖像與laplace結果相加
gray_plus_laplace = cv2.add(src_gray, diff_2)

# 灰度圖像與正規化後結果相加
gray_plus_std = cv2.add(src_gray, mean_f)

# 算 PSNR 值
psnr_value_sobel = calculate_psnr(src_gray, sobel)
psnr_value_laplacian = calculate_psnr(src_gray, laplacian)


print(f"PSNR Sobel: {psnr_value_sobel:.2f}")
print(f"PSNR Laplacian: {psnr_value_laplacian:.2f}")

# 顯示結果
cv2.imshow("First-order differential", diff_1)
cv2.imshow("sobel", gray_plus_sobel)
cv2.imshow("Mean Filter of first-order", mean_f)
cv2.imshow("std", gray_plus_std)
cv2.imshow("Second-order differential", diff_2)
cv2.imshow("laplace", gray_plus_laplace)
cv2.imshow("Unsharp Masking", src_img)

cv2.waitKey(0)
cv2.destroyAllWindows()


