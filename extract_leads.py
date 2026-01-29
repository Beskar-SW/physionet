import os
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

OUTPUT_DIR = "outputs/leads"
BBOX_DIR = "outputs/bounding_boxes"

MIN_ROW_PIXELS = 9  # mínimo de píxeles blancos para considerar señal
ROW_GAP = 10  # tolerancia entre filas
NUM_ROWS = 3  # usar solo las primeras 3 filas
LEADS_PER_ROW = 4

HSV_LOWER = np.array([0, 0, 0])
HSV_UPPER = np.array([180, 255, 150])

LEAD_COLORS = {
    "I":   (255, 0, 0),
    "II":  (0, 255, 0),
    "III": (0, 0, 255),
    "aVR": (255, 255, 0),
    "aVL": (255, 0, 255),
    "aVF": (0, 255, 255),
    "V1":  (128, 0, 0),
    "V2":  (0, 128, 0),
    "V3":  (0, 0, 128),
    "V4":  (128, 128, 0),
    "V5":  (128, 0, 128),
    "V6":  (0, 128, 128),
}

@dataclass(frozen=True)
class LeadConfig:
    image_path: str = "./test-0011.png"
    images_dir: str = "./test_images"
    output_dir: str = OUTPUT_DIR
    bbox_dir: str = BBOX_DIR
    min_row_pixels: int = MIN_ROW_PIXELS
    row_gap: int = ROW_GAP
    num_rows: int = NUM_ROWS
    leads_per_row: int = LEADS_PER_ROW
    hsv_lower: np.ndarray = HSV_LOWER
    hsv_upper: np.ndarray = HSV_UPPER
    show_debug: bool = True


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"No se pudo cargar {path}")
    return img


def crop_top_quarter(img):
    h, w = img.shape[:2]
    y_start = h // 4
    return img[y_start: (h//10) * 9, 0:w]

def build_mask(img, hsv_lower, hsv_upper):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, hsv_lower, hsv_upper)
    return mask


def detect_rows(binary, min_row_pixels, row_gap):
    row_sum = np.sum(binary > 0, axis=1)
    rows = np.where(row_sum > min_row_pixels)[0]

    if rows.size == 0:
        return []

    bands = []
    start = rows[0]
    prev = rows[0]

    for r in rows[1:]:
        if r <= prev + row_gap:
            prev = r
        else:
            bands.append((start, prev))
            start = r
            prev = r

    bands.append((start, prev))
    return bands


def split_columns(binary, row_band):
    y1, y2 = row_band
    _, w = binary.shape

    proportions = [0.278, 0.224, 0.224, 0.224]

    xs = [0]
    for p in proportions:
        xs.append(xs[-1] + int(p * w))

    xs[-1] = w

    leads = []
    for i in range(LEADS_PER_ROW):
        x1, x2 = xs[i], xs[i+1]
        crop = binary[y1:y2, x1:x2]
        leads.append((x1, y1, x2, y2, crop))

    return leads


def bounding_box(img, min_size=5):
    ys, xs = np.where(img > 0)
    if len(xs) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    if (x2 - x1) < min_size or (y2 - y1) < min_size:
        return None

    return x1, y1, x2, y2


def extract_leads(binary, output_dir, bbox_output_path, num_rows, min_row_pixels, row_gap, show_debug):
    debug_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    bands = detect_rows(binary, min_row_pixels, row_gap)

    if len(bands) < num_rows:
        raise RuntimeError("No se detectaron suficientes filas")

    bands = bands[:num_rows]

    lead_names = [
        ["I", "aVR", "V1", "V4"],
        ["II", "aVL", "V2", "V5"],
        ["III", "aVF", "V3", "V6"],
    ]

    metadata = []

    for row_idx, band in enumerate(bands):
        leads = split_columns(binary, band)

        for col_idx, (x1, y1, x2, y2, lead_img) in enumerate(leads):
            name = lead_names[row_idx][col_idx]

            bb = bounding_box(lead_img)
            if bb is None:
                print(f"[WARN] Sin señal en {name}")
                continue

            bx1, by1, bx2, by2 = bb
            lead_crop = lead_img[by1:by2, bx1:bx2]

            if lead_crop.size == 0:
                print(f"[WARN] Crop vacío en {name}")
                continue

            out_path = os.path.join(output_dir, f"{name}.png")
            cv2.imwrite(out_path, lead_crop)

            # Bounding box global
            gx1 = x1 + bx1
            gy1 = y1 + by1
            gx2 = x1 + bx2
            gy2 = y1 + by2

            color = LEAD_COLORS.get(name, (255, 255, 255))

            cv2.rectangle(debug_img, (gx1, gy1), (gx2, gy2), color, 2)
            cv2.putText(
                debug_img,
                name,
                (gx1, max(gy1 - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

            metadata.append({
                "lead": name,
                "row": row_idx,
                "bbox_global": (gx1, gy1, gx2, gy2),
                "path": out_path
            })

            print(f"[OK] {name} → {out_path}")

    if show_debug:
        cv2.imshow("ECG Bounding Boxes", resized_image(debug_img))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cv2.imwrite(bbox_output_path, debug_img)

    print("\nDerivaciones extraídas:")
    for m in metadata:
        print(m)

def resized_image(img):
    return cv2.resize(img, None, fx=0.5, fy=0.5)

def main():
    config = LeadConfig()
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.bbox_dir, exist_ok=True)

    images_dir = Path(config.images_dir)
    images = sorted(images_dir.glob("*.png")) if images_dir.exists() else []

    if not images:
        images = [Path(config.image_path)]

    for image_path in images:
        img = load_image(str(image_path))
        mask = build_mask(img, config.hsv_lower, config.hsv_upper)
        mask = crop_top_quarter(mask)

        cv2.imshow("Original", resized_image(mask))
        mask = cv2.GaussianBlur(mask, (3,3), 0)
        cv2.imshow("GaussianBlur", resized_image(mask))

        stem = image_path.stem
        leads_output_dir = os.path.join(config.output_dir, stem)
        os.makedirs(leads_output_dir, exist_ok=True)

        bbox_output_path = os.path.join(config.bbox_dir, f"{stem}_bounding_boxes.png")

        try:
            extract_leads(
                binary=mask,
                output_dir=leads_output_dir,
                bbox_output_path=bbox_output_path,
                num_rows=config.num_rows,
                min_row_pixels=config.min_row_pixels,
                row_gap=config.row_gap,
                show_debug=config.show_debug,
            )
        except:
            print(f"Error con la imagen {image_path}")
            continue


if __name__ == "__main__":
    main()
