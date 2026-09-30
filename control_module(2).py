import numpy as np

MAX_ANGLE = 1.0
MAX_ACCEL = 20.0     # tăng tốc lệnh (đơn vị / giây)

# ---- ĐƠN VỊ GÓC LÁI CỦA AVControl -------------------------------------------------------
# Toàn bộ bộ điều khiển bên trong chạy CHUẨN HOÁ trong [-1, 1] (MAX_ANGLE).
# AVControl thường nhận góc theo ĐỘ (vd. -25..25) chứ không phải -1..1, nên nếu gửi thẳng
# 1.0 thì bánh xe chỉ lệch ~1 độ -> xe gần như đi thẳng dù lệnh đã "hết lái".
# steer_to_angle() đổi sang đơn vị của AVControl ngay trước khi gửi.
#   STEER_SCALE : góc lái tối đa của xe (đơn vị AVControl). Nếu AVControl thật sự nhận -1..1 thì đặt 1.0
#   STEER_SIGN  : +1 nếu lệnh dương = quẹo phải; -1 nếu xe quẹo ngược
STEER_SCALE = 25.0
STEER_SIGN = 1.0
# Đường cong: angle = SCALE * |u|^STEER_CURVE. =1.0 là tuyến tính.
# >1 làm góc lái NHỎ hơn khi sai số nhỏ (đường thẳng đỡ lắc) nhưng vẫn hết lái khi |u| -> 1 (vào cua gắt).
STEER_CURVE = 1.5


def steer_to_angle(steering):
    """Đổi lái chuẩn hoá [-1, 1] -> góc gửi cho AVControl."""
    u = float(np.clip(steering, -1.0, 1.0))
    return float(np.sign(u) * (abs(u) ** STEER_CURVE) * STEER_SCALE * STEER_SIGN)


class PreviewSteering:
    """
    u = sf * k_near * e_near + sf * k_far * e_far + kd * d(e_near)/dt + I
      * sf   : giảm gain khi chạy nhanh (tránh dao động do độ trễ camera -> lái).
      * I    : số hạng tích phân RÒ (leaky), CHỈ tích luỹ khi đường THẬT SỰ cong (xem _curv trong compute()).
               Dùng để xoá độ lệch vĩnh viễn khi đi cua (không có nó, xe cân bằng lệch vào phía TRONG cua).
               Bật cả trên đường thẳng thì tích phân CỘNG DỒN với độ trễ camera->xe làm xe tự lắc chậm dần
               nặng hơn quanh tâm làn (chu kỳ vài giây, biên độ tăng dần) - đây là nguyên nhân xe "lắc khi
               lệch tâm" mà không phải do e_near/e_far tính sai.
      * kd   : hệ số đạo hàm được tăng (0.1 -> 0.5) để bù lại việc bớt tích phân trên đường thẳng - đạo hàm
               dập dao động ngay lập tức, không cần biết đường có cong hay không.
      * k_near/k_far : giảm (1.2/0.5 -> 0.8/0.3) vì tổ hợp gain cũ, cộng với trễ camera 0.2-0.4s thực tế,
               tự dao động ngay cả khi KHÔNG có nhiễu đo đạc - xem hàm steer_towards_wobble_demo trong
               báo cáo mô phỏng nếu cần dựng lại thử nghiệm.
      * e_*  : sai số được lọc thông thấp + vùng dead trước khi tính.
    """
    def __init__(self, k_near=0.8, k_far=0.3, kd=0.5, max_angle=MAX_ANGLE,
                 rate_per_sec=5.0, rate_boost_per_sec=30.0,
                 v_ref=15.0, min_gain=0.1, d_filter=0.85, gain_pow=1.0,
                 e_tau=0.08, e_tau_far=0.15, deadband=0.02,
                 ki=0.5, i_max=0.35, i_leak_tau=4.0,
                 i_curv_lo=0.08, i_curv_hi=0.25, i_leak_straight=0.7):
        # Bộ tham số chọn bằng mô phỏng xe khép kín (độ trễ 0.08-0.3s, 12-30 km/h, thang lái 15-40, cua R=25-70m)
        self.k_near, self.k_far, self.kd = k_near, k_far, kd
        self.max_angle, self.rate = max_angle, rate_per_sec
        self.rate_boost = rate_boost_per_sec   # tốc độ đổi THÊM khi sai số còn lại lớn
        self.v_ref, self.min_gain, self.d_filter = v_ref, min_gain, d_filter
        self.gain_pow = gain_pow    # độ lợi của xe ~ v^2 nên gain điều khiển giảm theo (v_ref/v)^2
        self.e_tau, self.e_tau_far = e_tau, e_tau_far   # hằng số thời gian lọc nhiễu (s); điểm xa nhiễu hơn
        self.deadband = deadband
        self.ki, self.i_max, self.i_leak_tau = ki, i_max, i_leak_tau
        # Tích phân CHỈ hoạt động khi đường thật sự cong (e_far lệch e_near). Đường thẳng: tích phân + độ trễ camera
        # làm xe dao động chậm quanh tâm (chu kỳ ~6-8 s) nên ở đó ép nó xả nhanh (i_leak_straight) và không tích luỹ.
        self.i_curv_lo, self.i_curv_hi, self.i_leak_straight = i_curv_lo, i_curv_hi, i_leak_straight
        self.reset()

    def reset(self):
        self.prev_e, self.d_f, self.out = None, 0.0, 0.0
        self.e_n = self.e_f = None
        self.i_s = 0.0

    def decay(self, factor=0.8):
        self.out *= factor
        self.i_s *= factor
        self.prev_e, self.d_f = None, 0.0
        self.e_n = self.e_f = None
        return float(self.out)

    def steer_towards(self, target, dt):
        """Đưa .out dần về 'target' bằng đúng cơ chế rate-limit co dãn như compute(), nhưng KHÔNG
        dùng e_near/e_far. Dùng khi mất vạch (ngã ba/tư) và ta có mục tiêu lái riêng (vd từ
        HeadingCorrector) thay vì chỉ decay() thẳng về 0 - vì decay về 0 giữ nguyên hướng đầu xe
        đang lệch (nếu vừa cua xong thì lệch), có thể làm xe lao thẳng vào lề."""
        dt = float(np.clip(dt, 0.005, 0.05))
        target = float(np.clip(target, -self.max_angle, self.max_angle))
        gap = abs(target - self.out)
        dyn_rate = self.rate + self.rate_boost * min(gap / self.max_angle, 1.0)
        step = dyn_rate * dt
        self.out += float(np.clip(target - self.out, -step, step))
        self.i_s *= (1.0 - dt / self.i_leak_tau)
        self.prev_e, self.d_f = None, self.d_f * 0.9
        return float(self.out)

    @staticmethod
    def _lp(old, new, dt, tau):
        """Lọc thông thấp theo thời gian (không phụ thuộc fps)."""
        if old is None:
            return new
        a = 1.0 - float(np.exp(-dt / tau))
        return old + a * (new - old)

    def _db(self, e):
        return 0.0 if abs(e) < self.deadband else e - np.sign(e) * self.deadband

    def compute(self, e_near, e_far, speed, dt):
        dt = float(np.clip(dt, 0.005, 0.05))
        self.e_n = self._lp(self.e_n, e_near, dt, self.e_tau)
        self.e_f = self._lp(self.e_f, e_far, dt, self.e_tau_far)
        en, ef = self._db(self.e_n), self._db(self.e_f)

        sf = float(np.clip((self.v_ref / max(speed, self.v_ref)) ** self.gain_pow, self.min_gain, 1.0))
        raw_d = 0.0 if self.prev_e is None else (self.e_n - self.prev_e) / dt
        self.prev_e = self.e_n
        self.d_f = self.d_filter * self.d_f + (1 - self.d_filter) * raw_d

        # tích phân rò + chống bão hoà (không tích lũy khi đã hết lái)
        curv = abs(self.e_f - self.e_n)
        w_c = float(np.clip((curv - self.i_curv_lo) / (self.i_curv_hi - self.i_curv_lo), 0.0, 1.0))
        tau = self.i_leak_straight + (self.i_leak_tau - self.i_leak_straight) * w_c
        self.i_s *= (1.0 - dt / tau)
        if abs(self.out) < 0.9 * self.max_angle:
            self.i_s = float(np.clip(self.i_s + w_c * sf * self.ki * en * dt, -self.i_max, self.i_max))

        u = sf * (self.k_near * en + self.k_far * ef) + self.kd * self.d_f + self.i_s
        u = float(np.clip(u, -self.max_angle, self.max_angle))

        # rate limit co dãn: sai số nhỏ đổi chậm (êm), sai số lớn đổi nhanh (kịp vào cua)
        gap = abs(u - self.out)
        dyn_rate = self.rate + self.rate_boost * min(gap / self.max_angle, 1.0)
        step = dyn_rate * dt
        self.out += float(np.clip(u - self.out, -step, step))
        return float(self.out)


# Khi điểm xem trước thứ i lệch 0.35 làn thì tốc độ tối đa là (gần -> xa):
CAP_AT_035 = (14.0, 20.0, 26.0, 32.0, 38.0)


def compute_speed_command(steering, dev, seen, vis, cmd_speed, real_speed, dt,
                          max_speed=55.0, cruise_speed=50.0, min_speed=12.0,
                          stop=False, vis_gain=14.0):
    """
    Tốc độ mục tiêu = min( theo 5 điểm xem trước , theo góc lái , theo tầm nhìn ).
      * Điểm xem trước i lệch (so với hướng đi thẳng) > 0.12 làn thì bắt đầu phanh; lệch 0.35 -> CAP_AT_035[i];
        lệch 0.9 -> min_speed. Điểm 5 (xa nhất) lệch là phanh sớm nhất và nhẹ nhất.
      * stop=True (sắp cán vạch liền): dừng hẳn (lệnh 0).
    Phanh: hạ lệnh NGAY; nếu tốc độ thật còn cao hơn mục tiêu thì hạ lệnh thấp hơn nữa. Ga tăng dần MAX_ACCEL.
    """
    if stop:
        return 0.0
    caps = [cruise_speed]
    for i in range(5):
        if i < len(seen) and seen[i]:
            c35 = min(CAP_AT_035[i], cruise_speed)
            caps.append(float(np.interp(abs(dev[i]), [0.12, 0.35, 0.90], [cruise_speed, c35, min_speed])))
    v_pts = min(caps)
    ang = float(np.clip((abs(steering) - 0.05) / 0.65, 0.0, 1.0))
    v_ang = min_speed + (cruise_speed - min_speed) * (1.0 - ang) ** 1.5
    v_vis = float(np.clip(vis_gain * vis, min_speed, cruise_speed))
    desired = float(np.clip(min(v_pts, v_ang, v_vis), min_speed, max_speed))

    if real_speed > desired + 3.0:
        # Tốc độ THẬT còn cao hơn mục tiêu: luôn hạ lệnh thấp hơn nữa để phanh thật sự
        # (trước đây nhánh này chỉ chạy đúng 1 khung, sau đó cmd == desired nên xe vẫn lao vào cua).
        cmd = max(0.0, desired - 0.6 * (real_speed - desired))
    elif desired < cmd_speed:
        cmd = desired
    else:
        cmd = min(desired, cmd_speed + MAX_ACCEL * dt)
    return float(np.clip(cmd, 0.0, max_speed))


def read_speed(state, fallback):
    """Đọc tốc độ THẬT từ GetStatus(); nếu không đọc được thì dùng tốc độ lệnh."""
    try:
        keys = ("speed", "Speed", "velocity", "Velocity", "current_speed", "cur_speed")
        if isinstance(state, dict):
            for k in keys:
                if k in state:
                    return abs(float(state[k]))
        else:
            for k in keys:
                if hasattr(state, k):
                    return abs(float(getattr(state, k)))
    except Exception:
        pass
    return fallback


class LineGuard:
    def __init__(self, hold=0.6, creep=3.0, clear_time=0.3, away_steer=0.6,
                 min_seen_to_trust=1, confirm_frames=3):
        self.hold, self.creep, self.clear_time, self.away = hold, creep, clear_time, away_steer
        self.min_seen_to_trust = min_seen_to_trust   # số điểm "seen" tối thiểu để tin tín hiệu active
        self.confirm_frames = confirm_frames         # số frame LIÊN TIẾP cần để tin active=True
        self.reset()

    def reset(self):
        self.state, self.t, self.clear, self.side = 0, 0.0, 0.0, 1
        self._active_streak = 0

    def update(self, active, side, dt, seen=None):
        # Gần như mất hết vạch kẻ đường (ngã tư) -> không tin 'active', bỏ qua hoàn toàn.
        if seen is not None and sum(1 for s in seen if s) < self.min_seen_to_trust:
            self.reset()
            return None

        if self.state == 0:
            # Cần XÁC NHẬN liên tiếp 'confirm_frames' khung hình mới được tin
            # (tránh 1 khung nhiễu làm dừng oan giữa khúc cua).
            self._active_streak = self._active_streak + 1 if active else 0
            if self._active_streak < self.confirm_frames:
                return None
            self.state, self.t, self.clear = 1, 0.0, 0.0
        if active:
            self.clear = 0.0
            self.side = side or self.side
        else:
            self.clear += dt
            if self.clear >= self.clear_time:
                self.state = 0
                return None
        steer = -self.side * self.away
        if self.state == 1:
            self.t += dt
            if self.t >= self.hold:
                self.state = 2
            return 0.0, steer
        return self.creep, steer


class IntersectionHandler:
    def __init__(self, min_seen_to_trust=1, near_points=2, steer_gate=0.5,
                 trigger_frames=3, hold_frames=25, clear_frames=5, straight_speed=25.0):
        self.min_seen_to_trust = min_seen_to_trust
        self.near_points = near_points      # số điểm GẦN nhất xét riêng (0..near_points-1 trong 'seen')
        self.steer_gate = steer_gate        # |steering| vượt mức này -> coi là đang cua thật, bỏ qua tín hiệu near_lost
        self.trigger_frames = trigger_frames
        self.hold_frames = hold_frames
        self.clear_frames = clear_frames
        self.straight_speed = straight_speed
        self.reset()

    def reset(self):
        self.lost_count, self.hold, self.clear = 0, 0, 0

    def update(self, seen, steering=0.0, dev=None):
        n_seen = sum(1 for s in seen if s) if seen else 0

        # Tín hiệu 1 (cũ): mất gần như toàn bộ vạch.
        total_lost = n_seen < self.min_seen_to_trust

        # Tín hiệu 2 (mới): mất hết các điểm GẦN nhất, đáng tin hơn vì gần đầu xe -> ít bị
        # nhiễu bởi vạch của đường giao cắt phía xa. Bỏ qua khi đang bẻ lái mạnh (cua gắt thật).
        near_lost = seen is not None and not any(seen[:self.near_points])
        if abs(steering) > self.steer_gate:
            near_lost = False

        # Tín hiệu 3: điểm gần nhất 'thấy' nhưng bị cô lập (2 điểm kế tiếp mất) và lệch lớn -> đó là vạch
        # cong của đường nhánh ở ngã ba, KHÔNG phải làn mình. Coi như mất vạch để không bẻ lái theo.
        isolated = False
        reliable = True
        if seen is not None and dev is not None and len(seen) >= 3:
            isolated = bool(seen[0] and not seen[1] and not seen[2] and abs(dev[0]) > 0.3)
            if abs(steering) > self.steer_gate:
                isolated = False
            # khung 'đáng tin' để THOÁT ngã ba: 2 điểm gần liên tiếp đều thấy và không lệch bất thường
            reliable = bool(seen[0] and seen[1] and abs(dev[0]) < 0.35 and abs(dev[1]) < 0.35)

        is_lost_frame = total_lost or near_lost or isolated

        if is_lost_frame:
            self.lost_count += 1
            self.clear = 0
        else:
            self.lost_count = 0
            self.clear = self.clear + 1 if reliable else 0

        if self.lost_count >= self.trigger_frames:
            self.hold = self.hold_frames
        elif self.hold > 0:
            self.hold -= 1
            if self.clear >= self.clear_frames:
                self.hold = 0
        return self.hold > 0

class IntersectionDrive:
    """
    Máy trạng thái lái qua ngã ba / ngã tư (chạy SAU IntersectionHandler.update()).

      NORMAL   : chưa vào ngã. LIÊN TỤC ước lượng (theta, y) = lệch HƯỚNG đầu xe so với đường và lệch
                 VỊ TRÍ ngang so với tâm làn: dự đoán bằng góc lái thật gửi xuống xe (mô hình xe đạp,
                 dùng steer_to_angle) và hiệu chỉnh bằng 2 hàng quét GẦN (hàng cuối cùng mất vạch
                 nên số đo lúc vào ngã luôn MỚI - không dùng hàng xa vì chúng mất vạch trước).
      STRAIGHT : mù vạch -> tiếp tục dự đoán bằng góc lái, tự đánh lái đưa (theta, y) về 0
                 (đưa xe về GIỮA đường, đầu xe THẲNG theo đường) rồi mới đi thẳng.
      FIX      : xe đang LẮC quá mức lúc vào ngã -> PHANH DỪNG, giữ bánh thẳng, đợi ổn định
                 rồi mới sang STRAIGHT để chạy tiếp (ga tăng dần).

    Đơn vị: chiều dài tính theo 'bề rộng làn' (cùng đơn vị với dev/e_near). Các hằng số hình học
    (gap_lw, la_lw, far_lw, l_lw) là ƯỚC LƯỢNG theo camera/xe - chỉnh nếu xe chỉnh thiếu/quá.
    """
    NORMAL, STRAIGHT, FIX = 0, 1, 2
    NAMES = {0: "NORMAL", 1: "STRAIGHT", 2: "FIX"}

    def __init__(self, window=0.6, swing_thresh=0.35, tv_thresh=2.0, cross_min=2, dead=0.1,
                 settle_time=0.35, max_fix_time=2.0, brake_rate=60.0, stop_speed=4.0,
                 straight_speed=25.0, straight_tol=0.03, lane_ok_dev=0.4,
                 gap_lw=0.57,    # khoảng cách mặt đất giữa hàng quét 1 và 2 / bề rộng làn
                 la_lw=1.14,     # khoảng cách trung bình tới 2 hàng gần / bề rộng làn
                 far_lw=3.5,     # khoảng cách trung bình tới các hàng xa / bề rộng làn
                 l_lw=0.77,      # chiều dài cơ sở xe / bề rộng làn
                 lane_m=3.5,     # bề rộng làn (m) - chỉ để đổi km/h sang 'làn/giây'
                 mem_alpha=0.25, u_max=0.6, v_min_est=5.0, corr=0.6):
        self.window, self.swing_thresh, self.tv_thresh = window, swing_thresh, tv_thresh
        self.cross_min, self.dead = cross_min, dead
        self.settle_time, self.max_fix_time = settle_time, max_fix_time
        self.brake_rate, self.stop_speed = brake_rate, stop_speed
        self.straight_speed, self.straight_tol = straight_speed, straight_tol
        self.lane_ok_dev = lane_ok_dev
        self.gap_lw, self.la_lw, self.far_lw, self.l_lw = gap_lw, la_lw, far_lw, l_lw
        self.lane_m, self.mem_alpha, self.u_max = lane_m, mem_alpha, u_max
        self.v_min_est = v_min_est
        self.corr = corr       # <1: chỉnh 'nhẹ tay' để sai số hằng số hình học không làm xe chỉnh quá đà
        self.reset()

    def reset(self):
        self.state = self.NORMAL
        self.hist = []            # [(t, steering)] trong 'window' giây gần nhất
        self.now = 0.0
        self.fix_t = self.settle = 0.0
        self.th = self.y = 0.0    # ước lượng: lệch hướng (rad), lệch vị trí ngang (làn); + = đường nằm bên PHẢI xe
        self.est_psi, self.est_e = 0.0, 0.0    # (chỉ để in log)
        self.wobble_metric = (0, 0.0, 0.0)   # (số lần đổi dấu, biên độ, tổng biến thiên) - để debug

    @property
    def name(self):
        return self.NAMES[self.state]

    def _is_wobbly(self):
        s = [v for _, v in self.hist]
        if len(s) < 3:
            return False
        tv = float(np.sum(np.abs(np.diff(s))))
        swing = float(max(s) - min(s))
        sig = [np.sign(v) for v in s if abs(v) >= self.dead]
        crossings = sum(1 for a, b in zip(sig, sig[1:]) if a != b)
        self.wobble_metric = (crossings, swing, tv)
        return (crossings >= self.cross_min and swing >= self.swing_thresh) or tv >= self.tv_thresh

    def _predict(self, steering, real_speed, dt):
        """Mô hình xe đạp theo góc lái THẬT gửi xuống xe: d(theta)/dt = -(v/L) tan(delta)."""
        v_lw = max(real_speed, 0.0) / 3.6 / self.lane_m           # làn / giây
        delta = np.radians(steer_to_angle(steering))
        self.th += -(v_lw / self.l_lw) * float(np.tan(delta)) * dt
        self.y += v_lw * float(np.tan(self.th)) * dt
        self.est_psi = self.th * (self.far_lw - self.la_lw)
        self.est_e = self.y + self.th * self.la_lw

    def update(self, is_inter, steering, steer_ctl, cur_speed, real_speed, dt, lane=None):
        """Trả về (steering, speed) khi đang xử lý ngã ba/tư; None nếu để vòng lặp chính điều khiển bình thường.
        lane = (dev, seen): dev/seen thô của 5 điểm xem trước."""
        self.now += dt
        # Vạch khung HIỆN TẠI đã thấy lại rõ và đáng tin (2 điểm gần đều thấy, lệch không quá lớn) ->
        # có làn THẬT để bám, không cần đi thẳng theo ước lượng nữa - dù bộ đếm 'is_inter' bên ngoài
        # (dựa trên các khung TRƯỚC) có thể chưa kịp tắt. Áp dụng ngay cả khi đường đang cong gắt: dev
        # lớn ở điểm XA không sao, chỉ cần 2 điểm GẦN đáng tin là đủ để controller thật bám đúng hướng.
        lane_ok = bool(lane is not None and lane[1][0] and lane[1][1]
                       and abs(lane[0][0]) < self.lane_ok_dev and abs(lane[0][1]) < self.lane_ok_dev)
        if self.state == self.STRAIGHT and lane_ok:
            self.state = self.NORMAL
            self.hist = []
            return None
        if self.state == self.NORMAL:
            if not is_inter:
                self.hist.append((self.now, float(steering)))
                self.hist = [h for h in self.hist if self.now - h[0] <= self.window]
                self._predict(steering, real_speed, dt)
                if lane is not None and lane[1][0] and lane[1][1] and abs(lane[0][0]) < 0.6:
                    dev = lane[0]
                    th_m = (dev[1] - dev[0]) / self.gap_lw              # độ dốc làn so với đầu xe
                    y_m = 0.5 * (dev[0] + dev[1]) - th_m * self.la_lw    # lệch vị trí tại xe
                    a = self.mem_alpha
                    self.th += a * (float(np.arctan(th_m)) - self.th)
                    self.y += a * (y_m - self.y)
                return None
            self.state = self.FIX if self._is_wobbly() else self.STRAIGHT
            self.fix_t = self.settle = 0.0

        if self.state == self.FIX:
            steer = steer_ctl.steer_towards(0.0, dt)      # bánh về thẳng, có giới hạn tốc độ đổi -> không giật
            speed = max(cur_speed - self.brake_rate * dt, 0.0)
            self._predict(steer, real_speed, dt)
            self.fix_t += dt
            ok = abs(steer) <= self.straight_tol and real_speed <= self.stop_speed
            self.settle = self.settle + dt if ok else 0.0
            if self.settle >= self.settle_time or self.fix_t >= self.max_fix_time:
                self.state = self.STRAIGHT            # đã ổn định -> chạy tiếp (ga tăng dần bên dưới)
            return steer, speed

        # STRAIGHT: mù vạch -> tự chỉnh về giữa đường bằng ước lượng, hết lệch mới đi thẳng thật sự
        if not is_inter or lane_ok:
            self.state = self.NORMAL
            self.hist = []
            return None
        # Vạch phía xa SAU ngã tư hiện ra -> số đo thật: kéo ước lượng về đúng (bù sai số mô hình)
        if lane is not None:
            dev, seen = lane
            far = [dev[i] for i in (3, 4) if seen[i] and abs(dev[i]) < 0.9]
            if far:
                self.y += self.mem_alpha * ((float(np.mean(far)) - self.th * self.far_lw) - self.y)
        sf = float(np.clip(steer_ctl.v_ref / max(real_speed, steer_ctl.v_ref), steer_ctl.min_gain, 1.0))
        e_n = self.y + self.th * self.la_lw
        e_f = self.y + self.th * self.far_lw
        u = self.corr * sf * (steer_ctl.k_near * e_n + steer_ctl.k_far * e_f)
        u = float(np.clip(u, -self.u_max, self.u_max))
        steer = steer_ctl.steer_towards(u, dt)
        self._predict(steer, real_speed, dt)
        err = abs(self.y) + abs(self.th) * self.far_lw
        target = self.straight_speed if err < 0.2 else 0.6 * self.straight_speed   # còn lệch nhiều thì chạy chậm
        speed = cur_speed + float(np.clip(target - cur_speed, -30.0 * dt, MAX_ACCEL * dt))
        return steer, speed