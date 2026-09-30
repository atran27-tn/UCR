import cv2
import numpy as np

TOP_RATIO = 0.47          # bỏ phần trên (trời/núi/đường chân trời)
POLY_TOP_L = 0.15         # đa giác ROI: đỉnh trái/phải
POLY_TOP_R = 0.85
TOPHAT_K = 15             # kích thước phần tử cấu trúc top-hat
MAX_FLAT = 4.5            # bỏ cụm quá "bẹt" ngang (bbox rộng/cao > 4.5)
H_RUN = 25                # đoạn nằm ngang liên tục dài hơn số px này bị cắt

# ---- NGƯỠNG THÍCH NGHI (tự co giãn theo ánh sáng / màu sơn của từng bản đồ) ----------
THRESH = 15               # SÀN của ngưỡng top-hat mạnh (không bao giờ nhạy hơn mức này -> không thêm nhiễu)
THRESH_MAX = 45           # trần ngưỡng top-hat mạnh (bản đồ rất sáng)
THRESH_K = 0.45           # ngưỡng mạnh = THRESH_K * percentile99(top-hat), kẹp trong [THRESH, THRESH_MAX]
WEAK_RATIO = 0.6          # ngưỡng YẾU = WEAK_RATIO * ngưỡng mạnh (để nối đoạn vạch mờ / ở xa)
SAT_MIN = 60              # trần độ bão hoà tối thiểu của vạch sơn
SAT_MIN_LO = 35           # sàn: bản đồ sơn nhạt màu vẫn bắt được, nhưng vẫn loại tuyết trắng (~10)
SAT_PCT = 98              # độ bão hoà tối thiểu = 0.5 * percentile98 của ROI, kẹp trong [SAT_MIN_LO, SAT_MIN]
CLAHE_CLIP = 2.0          # cân bằng tương phản cục bộ cho kênh sáng (bóng cây, nắng gắt, chiều tối)
# ---------------------------------------------------------------------------

_clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8, 8))


def _poly_mask(rh, rw):
    poly = np.array([
        [0, rh], [rw, rh],
        [int(rw * POLY_TOP_R), 0], [int(rw * POLY_TOP_L), 0]
    ], dtype=np.int32)
    m = np.zeros((rh, rw), np.uint8)
    cv2.fillPoly(m, [poly], 255)
    return m


def crop_roi(raw_image, apply_poly=True):
    h = raw_image.shape[0]
    roi = raw_image[int(h * TOP_RATIO):, :].copy()
    if apply_poly:
        roi = cv2.bitwise_and(roi, roi, mask=_poly_mask(*roi.shape[:2]))
    return roi


def _adaptive_thresholds(tophat, sat, poly_mask):
    """Ước lượng ngưỡng từ chính khung hình (lấy mẫu thưa 1/16 điểm cho nhanh)."""
    t_s, s_s = tophat[::4, ::4], sat[::4, ::4]
    if poly_mask is not None:
        v = poly_mask[::4, ::4] > 0
        t_s, s_s = t_s[v], s_s[v]
    else:
        t_s, s_s = t_s.ravel(), s_s.ravel()
    if t_s.size < 200:
        return float(THRESH), float(SAT_MIN)
    thr = float(np.clip(THRESH_K * np.percentile(t_s, 99), THRESH, THRESH_MAX))
    sat_min = float(np.clip(0.5 * np.percentile(s_s, SAT_PCT), SAT_MIN_LO, SAT_MIN))
    return thr, sat_min


def make_lane_mask(roi, poly_mask=None):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat, val = hsv[..., 1], hsv[..., 2]
    val = cv2.GaussianBlur(_clahe.apply(val), (3, 3), 0)

    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (TOPHAT_K, TOPHAT_K))
    tophat = cv2.morphologyEx(val, cv2.MORPH_TOPHAT, kern)

    thr_hi, sat_hi = _adaptive_thresholds(tophat, sat, poly_mask)
    thr_lo, sat_lo = WEAK_RATIO * thr_hi, 0.8 * sat_hi

    # Ngưỡng kép (hysteresis): 'yếu' để lấy trọn vạch mờ, 'mạnh' để chắc chắn đó là vạch.
    # Cụm yếu chỉ được giữ nếu chứa ít nhất 1 điểm mạnh -> vạch xa/mờ liền nét, nhiễu đơn lẻ bị loại.
    weak = ((tophat > thr_lo) & (sat > sat_lo)).astype(np.uint8) * 255
    strong = (tophat > thr_hi) & (sat > sat_hi)

    # cắt các đoạn NGANG dài (vạch thật luôn nghiêng/đứng nên không bị ảnh hưởng)
    horiz = cv2.morphologyEx(weak, cv2.MORPH_OPEN, np.ones((1, H_RUN), np.uint8))
    horiz = cv2.dilate(horiz, np.ones((3, 3), np.uint8))
    weak = cv2.bitwise_and(weak, cv2.bitwise_not(horiz))
    strong &= weak > 0

    n, labels, stats, _ = cv2.connectedComponentsWithStats(weak, connectivity=8)
    if n > 1:
        bw = stats[:, cv2.CC_STAT_WIDTH].astype(np.float32)
        bh = stats[:, cv2.CC_STAT_HEIGHT].astype(np.float32)
        area = stats[:, cv2.CC_STAT_AREA]
        long_s = np.maximum(bw, bh)
        has_strong = np.bincount(labels[strong], minlength=n) > 0
        keep = (area >= 8) & (long_s >= 5) & (area < 4000) & (bw < MAX_FLAT * bh) & has_strong
        keep[0] = False
        m = np.where(keep, 255, 0).astype(np.uint8)[labels]
    else:
        m = np.zeros_like(weak)

    # Nối các khe nhỏ theo chiều dọc (vạch đứt vẫn đứt, chỉ vá lỗ do nhiễu) + làm đầy vạch
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 3), np.uint8))

    if poly_mask is not None:
        m = cv2.bitwise_and(m, poly_mask)
    return m


def get_lane_mask(raw_image):
    full = crop_roi(raw_image, apply_poly=False)
    poly = _poly_mask(*full.shape[:2])
    lane_mask = make_lane_mask(full, poly)
    roi_view = cv2.bitwise_and(full, full, mask=poly)   # chỉ để hiển thị "Cuted ROI"
    return lane_mask, roi_view