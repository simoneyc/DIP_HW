import numpy as np
import cv2

def dilate(img_matrix=None, kernel=np.ones((3, 3))):
    img_matrix = np.asarray(img_matrix)
    shape = np.asarray(kernel).shape
    dilated_img = np.zeros((img_matrix.shape[0], img_matrix.shape[1]), dtype=img_matrix.dtype)
    origin = (int((kernel.shape[0] - 1) / 2), int((kernel.shape[1] - 1) / 2))

    for i in range(len(img_matrix)):
        for j in range(len(img_matrix[0])):
            overlap = img_matrix[check(i - origin[0]):i + (shape[0] - origin[0]),check(j - origin[1]):j + (shape[1] - origin[1])]
            shp = overlap.shape

            first_row = int(np.fabs(i - origin[0])) if i - origin[0] < 0 else 0
            first_col = int(np.fabs(j - origin[1])) if j - origin[1] < 0 else 0

            last_row = shape[0] - 1 - (i + (shape[0] - origin[0]) - img_matrix.shape[0]) if i + (shape[0] - origin[0]) > img_matrix.shape[0] else shape[0] - 1
            last_col = shape[1] - 1 - (j + (shape[1] - origin[1]) - img_matrix.shape[1]) if j + (shape[1] - origin[1]) > img_matrix.shape[1] else shape[1] - 1

            if shp[0] != 0 and shp[1] != 0 and np.logical_and(kernel[first_row:last_row + 1, first_col:last_col + 1],overlap).any():
                dilated_img[i, j] = 255

    return dilated_img


def erode(img_matrix=None, kernel=np.ones((3, 3))):
    img_matrix = np.asarray(img_matrix)
    shape = np.asarray(kernel).shape
    eroded_img = np.zeros((img_matrix.shape[0], img_matrix.shape[1]), dtype=img_matrix.dtype)
    origin = (int(np.ceil((kernel.shape[0] - 1) / 2.0)), int(np.ceil((kernel.shape[1] - 1) / 2.0)))
    for i in range(len(img_matrix)):
        for j in range(len(img_matrix[0])):
            overlap = img_matrix[check(i - origin[0]):i + (shape[0] - origin[0]),check(j - origin[1]):j + (shape[1] - origin[1])]
            shp = overlap.shape
            first_row = int(np.fabs(i - origin[0])) if i - origin[0] < 0 else 0
            first_col = int(np.fabs(j - origin[1])) if j - origin[1] < 0 else 0

            last_row = shape[0] - 1 - (i + (shape[0] - origin[0]) - img_matrix.shape[0]) if i + (shape[0] - origin[0]) > img_matrix.shape[0] else shape[0] - 1
            last_col = shape[1] - 1 - (j + (shape[1] - origin[1]) - img_matrix.shape[1]) if j + (shape[1] - origin[1]) > img_matrix.shape[1] else shape[1] - 1

            if shp[0] != 0 and shp[1] != 0 and np.array_equal(np.logical_and(overlap, kernel[first_row:last_row + 1,first_col:last_col + 1]),kernel[first_row:last_row + 1,first_col:last_col + 1]):
                eroded_img[i, j] = 255

    return eroded_img

def check(index):
    return 0 if index < 0 else index


def boundary_extraction():
    
    img_path = "pika.jpg"
    input_img = cv2.imread(img_path)

    # 調整圖像大小
    # height, width = input_img.shape[:2]
    # new_dimensions = (width // 2, height // 2)
    # input_img = cv2.resize(input_img, new_dimensions, interpolation=cv2.INTER_AREA)

    cv2.imshow("input", input_img)

    gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    # 二值化
    thresh = 128
    img_binary = cv2.threshold(gray_img, thresh, 255, cv2.THRESH_BINARY)[1]
    origin_el = np.ones((3, 3))

    # Erosion
    erosion = erode(img_binary, origin_el)
    # Dilation
    dilation = dilate(img_binary, origin_el)

    # 膨脹 - 腐蝕 = 邊界
    boundary = dilation - erosion
    # 原圖 - 腐蝕 = 邊界
    # boundary = img_binary - erosion
    # 膨脹 - 原圖 = 邊界
    # boundary = dilation - img_binary
    # 二值圖畫素取反
    result = 255 - boundary

    cv2.imshow("erode", erosion)
    cv2.imshow("dilate", dilation)
    cv2.imshow("boundary", result)

    # 結束程式
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def region_filling():
    
    img_path = "cell.png"
    input_img = cv2.imread(img_path)

    # 調整圖像大小
    height, width = input_img.shape[:2]
    new_dimensions = (width // 3, height // 3)
    input_img = cv2.resize(input_img, new_dimensions, interpolation=cv2.INTER_AREA)

    cv2.imshow("input", input_img)

    gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    # 二值化
    thresh = 128
    img_binary = cv2.threshold(gray_img, thresh, 255, cv2.THRESH_BINARY)[1]

    # 構造Marker圖像
    marker = np.zeros_like(img_binary)
    marker[0, :] = 255
    marker[-1, :] = 255
    marker[:, 0] = 255
    marker[:, -1] = 255

    mask = 255 - img_binary
    cv2.imshow("mask", mask)
    origin_el = np.ones((3, 3))

    while True:
        marker_pre = marker.copy()
        dilation = dilate(marker, origin_el)
        marker = np.minimum(dilation, mask)

        if np.array_equal(marker_pre, marker):
            break

    result = 255 - marker
    cv2.imshow("result", result)

    filling = result - img_binary
    cv2.imshow("region filling", filling)

    # 結束程式
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    boundary_extraction()
    region_filling()
