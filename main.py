import cv2
import sys
import numpy as np

def mostrar_imagen(imagen, titulo,scale = 0.5):
    img_scaled = cv2.resize(imagen, None, fx=scale,fy=scale)
    cv2.imshow(titulo, img_scaled)

image_path = "./test_images/test-0002.png"
scale_by = 0.5

img = cv2.imread(image_path)

if img is None:
    sys.exit(1)

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower = np.array([0, 0, 0])
upper = np.array([180, 255, 150])
mask = cv2.inRange(hsv_img, lower, upper)  # Se genera la imagen binaria en blanco y negro
mask_norm = mask.astype(np.float32) / 255.0 #normalizar los datos para tener [0-1] en sus valores
mostrar_imagen(mask, "Imagen original binaria")

## Eliminar cuadricula

# lsd = cv2.createLineSegmentDetector(0)
# lines = lsd.detect(mask)[0]
# mask_lines = np.zeros(mask.shape, dtype=np.uint8)
#
# if lines is not None:
#     for line in lines:
#         x0, y0, x1, y1 = line.flatten().astype(int)
#         cv2.line(mask_lines, (x0, y0), (x1, y1), 255, thickness=2)
#
# mask_lines = cv2.GaussianBlur(mask_lines, (3,3), 0)
# mostrar_imagen(mask_lines, "Mascara lineas")

mask_lines = np.zeros(mask.shape, dtype=np.uint8)
test = cv2.Canny(mask, 900, 950)
lines = cv2.HoughLinesP(test, 0.5, np.pi/180, threshold=15, minLineLength=50, maxLineGap=20)
tolerancia = 2
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Calcular el ángulo en grados
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)

        # Filtro:
        es_horizontal = (angle < tolerancia) or (angle > 180 - tolerancia)
        es_vertical = (abs(angle - 90) < tolerancia)

        if es_horizontal or es_vertical:
            cv2.line(mask_lines, (x1, y1), (x2, y2), 255, 2)

        # cv2.line(mask_lines, (x1, y1), (x2, y2), 255, 2)

mostrar_imagen(mask_lines, "Lineas Hough")

## Limpieza de la imagen mediante umbralizacion de densidad
umbral_densidad = 0.17
window_size = 7

kernel = np.ones( (window_size, window_size), np.float32 )
mapa_densidad = cv2.filter2D(mask_norm, -1, kernel)

max_puntos = window_size * window_size
densidad_normalizada = mapa_densidad / max_puntos
clean_mask = np.where(densidad_normalizada >= umbral_densidad, 255, 0).astype(np.uint8)
clean_mask = cv2.bitwise_and(mask, clean_mask)
# mostrar_imagen(clean_mask, "Imagen binaria limpia")

cv2.waitKey(0)

cv2.imwrite(f"{image_path.split('/')[-1].split('.')[0]}_binary.png", clean_mask)

cv2.destroyAllWindows()
