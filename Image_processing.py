import cv2
import numpy as np

def parallel_histogram_equalization(src, num_threads):
    src_yuv = cv2.cvtColor(src, cv2.COLOR_BGR2YUV)
    yuv_channels = list(cv2.split(src_yuv))
    equalized_y = yuv_channels[0].copy()

    height, width = src.shape[:2]
    for y in range(height):
        for x in range(width):
            equalized_y[y, x] = np.clip(equalized_y[y, x], 0, 255)

    equalized_y = cv2.equalizeHist(equalized_y)
    yuv_channels[0] = equalized_y

    src_yuv = cv2.merge(yuv_channels)
    dst = cv2.cvtColor(src_yuv, cv2.COLOR_YUV2BGR)
    return dst

def color_correction(img):
    corrected_img = img.copy()
    corrected_img[:, :, 0] = np.minimum(corrected_img[:, :, 0] * 1.2, 255.0)
    corrected_img[:, :, 1] = np.minimum(corrected_img[:, :, 1] * 1.1, 255.0)
    corrected_img[:, :, 2] = np.minimum(corrected_img[:, :, 2] * 0.9, 255.0)
    return corrected_img.astype(np.uint8)

def parallel_resize(img, new_width, new_height):
    return cv2.resize(img, (new_width, new_height))

def blur(img):
    return cv2.blur(img, (5, 5))

def apply_filter(img, kernel):
    return cv2.filter2D(img, -1, kernel)

def parallel_rotate(src, angle):
    center = (src.shape[1] // 2, src.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    dst = cv2.warpAffine(src, rotation_matrix, (src.shape[1], src.shape[0]), flags=cv2.INTER_LINEAR)
    return dst

def main():
    img = cv2.imread("example2.jpeg")

    if img is None:
        print("Error: Image not found.")
        return

    num_threads = 4

    dst = parallel_histogram_equalization(img, num_threads)

    corrected_img = color_correction(img)
    resized_img = parallel_resize(img, 800, 800)
    blurred_img = blur(img)

    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]], dtype=np.float32)
    filtered_img = apply_filter(img, kernel)
    image_rotation = parallel_rotate(img, 90.0)

    cv2.imwrite("corrected_image.jpg", corrected_img)
    cv2.imwrite("resized_image.jpg", resized_img)
    cv2.imwrite("blurred_image.jpg", blurred_img)
    cv2.imwrite("filtered_image.jpg", filtered_img)
    cv2.imwrite("histogram_clache.jpg", dst)
    cv2.imwrite("rotated_image.jpg", image_rotation)

    cv2.imshow("Original Image", img)
    cv2.imshow("Color Corrected Image", corrected_img)
    cv2.imshow("Resized Image", resized_img)
    cv2.imshow("Blurred Image", blurred_img)
    cv2.imshow("Filtered Image", filtered_img)
    cv2.imshow("histogram_clache.jpg", dst)
    cv2.imshow("Rotated Image", image_rotation)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
