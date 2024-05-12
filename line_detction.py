import numpy as np
import matplotlib.pyplot as plt
import cv2

def canny(img):
    img = cv2.blur(img, (5,5))
    img = cv2.Canny(img, 100, 150)
    return img

def get_triangle_mask(img):
    h, w = img.shape
    mask = np.zeros_like(img)
    polygons = np.array([
        [(0, h-40), (w, h-40), (500, 400)]
    ])
    cv2.fillPoly(mask, polygons, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def display_lines(img, lines:np.ndarray):
    like_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            if x2 - x1 > 500:
                continue
            cv2.line(like_img, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return like_img

img_path = "road.jpeg"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

canny_img = canny(img)
masked_img = get_triangle_mask(canny_img)
lines = cv2.HoughLinesP(masked_img, 1, np.pi/360, 300, np.array([]), minLineLength=100, maxLineGap=10)
line_image = display_lines(img, lines)
combo_img = cv2.addWeighted(line_image, 0.8, img, 1, 1)

plt.imshow(combo_img, cmap="gray")
plt.show()