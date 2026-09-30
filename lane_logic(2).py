from dataclasses import dataclass, field
import numpy as np

N_POINTS = 5


@dataclass
class LaneResult:
    valid: bool = False
    e_near: float = 0.0
    e_far: float = 0.0
    dev: list = field(default_factory=lambda: [0.0] * N_POINTS)
    seen: list = field(default_factory=lambda: [False] * N_POINTS)
    vis: float = 0.0
    guard: object = None
    centers: list = field(default_factory=list)
    points: list = field(default_factory=list)
    lines: list = field(default_factory=list)   # [(x_vạch_trái | None, x_vạch_phải | None, y)] mỗi hàng quét
    xc: float = 0.0                               # x (pixel) được coi là trục xe

    @property
    def n_seen(self):
        return sum(1 for s in self.seen if s)


class LaneDetector:
    def __init__(self, lane_pos=0.5, car_x=0.5,
                 rows=(0.90, 0.74, 0.58, 0.44, 0.32),          # vị trí 5 hàng quét (0 = mép trên ROI, 1 = mép dưới)
                 lane_w_frac=(0.62, 0.52, 0.43, 0.36, 0.30),   # bề rộng làn ban đầu / rộng ảnh (sẽ tự học lại)
                 band_frac=0.03,          # nửa chiều cao dải quét / cao ROI
                 min_line_px=2,           # vạch mỏng hơn thế -> coi là nhiễu
                 max_line_frac=0.12,      # vạch rộng hơn thế (vạch dừng, vạch sang đường...) -> bỏ
                 guard_dist=0.22,         # vạch liền cách tâm xe < 0.22 làn -> báo guard
                 solid_ratio=0.8,         # >= 80% chiều dài vạch có pixel -> vạch LIỀN
                 w_alpha=0.1,             # hệ số làm mượt bề rộng làn
                 vis_weights=(0.6, 0.7, 0.8, 0.9, 1.0)):
        self.lane_pos = float(lane_pos)
        self.car_x = float(car_x)        # trục xe nằm ở đâu trong ảnh (0.5 = giữa). Camera lệch trục xe thì chỉnh số này
        self.rows = rows
        self.lane_w_frac = lane_w_frac
        self.band_frac = band_frac
        self.min_line_px = min_line_px
        self.max_line_frac = max_line_frac
        self.guard_dist = guard_dist
        self.solid_ratio = solid_ratio
        self.w_alpha = w_alpha
        self.vis_weights = vis_weights
        self.reset()

    def reset(self):
        self.wpx = [None] * N_POINTS      # bề rộng làn (pixel) đã học, theo từng hàng
        self._last_mid = None             # tâm làn ở hàng gần nhất, khung trước
        self._miss = 0                    # số khung liên tiếp mất hàng gần nhất

    # ------------------------------------------------------------------ tiện ích
    @staticmethod
    def _prep(mask):
        if mask is None:
            return None
        if mask.ndim == 3:
            mask = mask.max(axis=2)
        return mask > 0

    def _candidates(self, band, W):
        """Tìm các vạch (x tâm) trong một dải ngang."""
        fill = max(1, int(0.2 * band.shape[0]))
        idx = np.flatnonzero(band.sum(axis=0) >= fill)
        if idx.size == 0:
            return []
        out = []
        for g in np.split(idx, np.flatnonzero(np.diff(idx) > 3) + 1):
            w = g[-1] - g[0] + 1
            if w < self.min_line_px or w > self.max_line_frac * W:
                continue
            out.append(float(g[0] + g[-1]) / 2.0)
        return out

    @staticmethod
    def _match(cands, ref, wpx, learned):
        """Ghép vạch thành làn. Ưu tiên cặp (trái, phải); không có thì dùng 1 vạch + bề rộng làn."""
        tol = 0.25 if learned else 0.45
        best, best_cost = None, 1e9
        for a in range(len(cands)):
            for b in range(a + 1, len(cands)):
                sep = cands[b] - cands[a]
                if abs(sep - wpx) > tol * wpx:
                    continue
                mid = 0.5 * (cands[a] + cands[b])
                off = abs(mid - ref)
                if off > 0.7 * wpx:
                    continue
                cost = off + abs(sep - wpx)
                if cost < best_cost:
                    best_cost = cost
                    best = dict(mid=mid, wpx=sep, left=cands[a], right=cands[b], pair=True)
        if best is not None:
            return best
        best_cost = 1e9
        for c in cands:
            for mid, is_left in ((c + wpx / 2.0, True), (c - wpx / 2.0, False)):
                off = abs(mid - ref)
                if off > 0.45 * wpx or off >= best_cost:
                    continue
                best_cost = off
                best = dict(mid=mid, wpx=wpx, left=c if is_left else None,
                            right=None if is_left else c, pair=False)
        return best

    def _is_solid(self, mask, pts, H, W):
        """Vạch có liền nét không: dọc theo đường thẳng đi qua các điểm đã tìm thấy, tỉ lệ hàng có pixel."""
        xs = np.array([p[0] for p in pts], float)
        ys = np.array([p[1] for p in pts], float)
        samples = np.linspace(self.rows[3] * (H - 1), self.rows[0] * (H - 1), 16)
        if len(pts) >= 2 and np.ptp(ys) > 1:
            k, b = np.polyfit(ys, xs, 1)
            xs_s = k * samples + b
        else:
            xs_s = np.full_like(samples, xs[0])
        r = max(3, int(0.02 * W))
        hit = 0
        for x, y in zip(xs_s, samples):
            yi, xi = int(round(y)), int(round(x))
            if 0 <= yi < H and mask[yi, max(0, xi - r):min(W, xi + r + 1)].any():
                hit += 1
        return hit / len(samples) >= self.solid_ratio

    # ------------------------------------------------------------------ chính
    def compute(self, lane_mask):
        mask = self._prep(lane_mask)
        res = LaneResult()
        if mask is None or mask.size == 0:
            res.points = [(0.0, 0.0, 1.0, False)] * N_POINTS
            return res

        H, W = mask.shape
        xc = self.car_x * W
        bh = max(3, int(self.band_frac * H))
        ref = self._last_mid if (self._last_mid is not None and self._miss < 10) else xc

        dev, seen, points, centers, lines = [], [], [], [], []
        left_pts, right_pts = [], []          # (x, y, wpx, chỉ số hàng)
        mids = []

        for i in range(N_POINTS):
            y = int(np.clip(self.rows[i] * (H - 1), 0, H - 1))
            band = mask[max(0, y - bh):min(H, y + bh + 1)]
            wpx = self.wpx[i] if self.wpx[i] else self.lane_w_frac[i] * W
            m = self._match(self._candidates(band, W), ref, wpx, self.wpx[i] is not None)

            if m is None:
                seen.append(False)
                dev.append(0.0)
                mids.append(None)
                lines.append((None, None, float(y)))
                points.append((float(np.clip(ref, 0, W - 1)), float(y), float(wpx), False))
                continue

            w = float(m["wpx"])
            if m["pair"]:
                old = self.wpx[i]
                self.wpx[i] = w if old is None else (1 - self.w_alpha) * old + self.w_alpha * w
            tx = m["mid"] + (self.lane_pos - 0.5) * w        # x đích của điểm i
            seen.append(True)
            dev.append(float((tx - xc) / w))
            mids.append(m["mid"])
            points.append((float(tx), float(y), w, True))
            centers.append((float(tx), float(y)))
            ref = m["mid"]
            lines.append((m["left"], m["right"], float(y)))
            if m["left"] is not None:
                left_pts.append((m["left"], y, w, i))
            if m["right"] is not None:
                right_pts.append((m["right"], y, w, i))

        # theo dõi hàng gần nhất giữa các khung
        if mids[0] is not None:
            self._last_mid, self._miss = mids[0], 0
        else:
            self._miss += 1

        # e_near / e_far
        near = [dev[i] for i in (0, 1) if seen[i]]
        far = [dev[i] for i in (2, 3, 4) if seen[i]]
        if near:
            e_near = float(np.mean(near))
        elif far:
            e_near = float(far[0])
        else:
            e_near = 0.0
        e_far = float(np.mean(far)) if far else e_near

        # vis: cộng trọng số các điểm thấy liên tiếp từ gần ra xa
        vis = 0.0
        for i in range(N_POINTS):
            if not seen[i]:
                break
            vis += self.vis_weights[i]

        # guard: vạch LIỀN bên nào đang sát tâm xe
        guard, best_d = None, self.guard_dist
        for side, pts in (("right", right_pts), ("left", left_pts)):
            if not pts or pts[0][3] > 1:            # chỉ xét khi thấy vạch ở 2 hàng gần nhất
                continue
            x, _, w, _ = pts[0]
            d = ((x - xc) if side == "right" else (xc - x)) / w
            if d < best_d and self._is_solid(mask, pts, H, W):
                guard, best_d = side, d

        res.valid = any(seen[:3])
        res.e_near, res.e_far = e_near, e_far
        res.dev, res.seen, res.vis = dev, seen, vis
        res.guard = guard
        res.centers, res.points = centers, points
        res.lines, res.xc = lines, float(xc)
        return res