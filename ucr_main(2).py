# -*- coding: utf-8 -*-
from ucr_lib import GetStatus, GetRaw, AVControl, CloseSocket
import cv2
import numpy as np
import time

from preprocessing import get_lane_mask
from lane_logic import LaneDetector
from control_module import (PreviewSteering, compute_speed_command, read_speed,
                            LineGuard, IntersectionHandler, IntersectionDrive, steer_to_angle)

# --- CẤU HÌNH ---
TARGET_FPS = 60.0
FRAME_TIME = 1.0 / TARGET_FPS
CRUISE_SPEED = 35.0     # đường thẳng. Tăng dần 35 -> 45 -> 50 khi đã chạy ổn
MAX_SPEED = 55.0
MIN_SPEED = 12.0
CAR_X_FRAC = 0.5        # trục xe nằm ở đâu trong ảnh camera. Xe chạy sát mép PHẢI dù near~0 -> tăng số này (0.55, 0.6...)
LANE_POS = 0.5          # 0.5 = giữa làn phải ; 0.0 = chạy dọc vạch đứt (giữa cả đường)
GUARD_SIDES = ("right",)   # sắp cán vạch LIỀN bên nào thì dừng. Muốn cả bên trái: ("right", "left")
GUARD_AWAY = 0.6        # góc lái đánh thêm để tránh xa vạch khi guard bật
CREEP_SPEED = 3.0       # nhích chậm để thoát vạch nếu đã dừng mà vạch vẫn còn quá gần

# Bao nhiêu khung hình LIÊN TIẾP phải báo "có vạch liền" mới được tin và dừng xe
# (tránh 1 khung nhiễu như bóng cây/tuyết làm dừng oan giữa khúc cua).
GUARD_CONFIRM_FRAMES = 3
# Ít nhất bao nhiêu / 5 điểm xem trước phải "seen" thì tín hiệu guard mới đáng tin
# (ở ngã ba/ngã tư mất vạch kẻ đường thì guard hay báo nhiễu -> bỏ qua).
GUARD_MIN_SEEN = 2
# Đang bẻ lái mạnh (vào cua) thì KHÔNG tin guard: vào cua phải thì vạch liền bên phải tự nhiên
# áp sát tâm xe, guard sẽ đánh lái ngược (sang TRÁI) + dừng xe -> văng ra lề trái.
GUARD_MAX_STEER = 0.35
DEBUG_PRINT_EVERY = 0.25   # giây; 0 = tắt in log

# Ngã ba / ngã tư: mất hết vạch >= INTER_TRIGGER_FRAMES khung -> đi thẳng, giữ tốc độ vừa phải.
INTER_TRIGGER_FRAMES = 3
INTER_HOLD_FRAMES = 130     # ~2.3 s ở 56 fps: đủ để băng qua ngã tư dài
INTER_CLEAR_FRAMES = 5
INTER_STRAIGHT_SPEED = 25.0
# FIX ngã ba/tư NHỎ: nếu cả INTER_NEAR_POINTS điểm GẦN nhất đều mất vạch thì cũng coi là ngã ba/tư,
# dù vài điểm XA vẫn còn thấy nhiễu (vạch của đường giao cắt) - không cần đợi mất hết cả 5 điểm.
INTER_NEAR_POINTS = 2
# Nhưng nếu đang bẻ lái mạnh hơn mức này (đang vào cua gắt thật sự) thì bỏ qua tín hiệu "mất điểm
# gần" ở trên, vì lúc cua gắt điểm gần có thể tạm mất do góc camera chứ không phải ngã ba/tư.
INTER_STEER_GATE = 0.5
# Vào ngã ba/tư: lái về THẲNG rồi đi thẳng. Nếu xe đang LẮC (góc lái đổi dấu liên tục trước khi vào ngã)
# thì DỪNG hẳn, chỉnh bánh thẳng, đợi ổn định rồi mới chạy tiếp.
WOBBLE_WINDOW = 0.6          # giây xét lịch sử góc lái trước khi vào ngã
WOBBLE_SWING = 0.35          # biên độ lắc tối thiểu (thang -1..1) để tính là lắc
WOBBLE_CROSSINGS = 2         # số lần đổi dấu tối thiểu trong cửa sổ
WOBBLE_TV = 2.0              # hoặc tổng biến thiên góc lái vượt mức này
INTER_SETTLE_TIME = 0.35     # giây bánh thẳng + xe đã dừng thì mới chạy tiếp
INTER_STOP_SPEED = 4.0       # tốc độ thật dưới mức này coi như đã dừng


def main():
    print("[UCR 2026] Khởi động hệ thống...")

    lane_detector = LaneDetector(lane_pos=LANE_POS, car_x=CAR_X_FRAC)
    steer_ctl = PreviewSteering()      # mặc định đã chỉnh bằng mô phỏng (xem control_module.py)
    line_guard = LineGuard(hold=0.6, creep=CREEP_SPEED, clear_time=0.3, away_steer=GUARD_AWAY,
                           min_seen_to_trust=GUARD_MIN_SEEN, confirm_frames=GUARD_CONFIRM_FRAMES)
    inter_guard = IntersectionHandler(min_seen_to_trust=1, near_points=INTER_NEAR_POINTS,
                                      steer_gate=INTER_STEER_GATE, trigger_frames=INTER_TRIGGER_FRAMES,
                                      hold_frames=INTER_HOLD_FRAMES, clear_frames=INTER_CLEAR_FRAMES,
                                      straight_speed=INTER_STRAIGHT_SPEED)

    inter_drive = IntersectionDrive(window=WOBBLE_WINDOW, swing_thresh=WOBBLE_SWING, tv_thresh=WOBBLE_TV,
                                    cross_min=WOBBLE_CROSSINGS, settle_time=INTER_SETTLE_TIME,
                                    stop_speed=INTER_STOP_SPEED, straight_speed=INTER_STRAIGHT_SPEED)

    current_speed = 0.0
    real_speed = 0.0
    steering = 0.0
    printed_state = False
    last_print = 0.0
    last_frame_time = time.perf_counter()

    try:
        while True:
            frame_start = time.perf_counter()
            dt = float(np.clip(frame_start - last_frame_time, 0.005, 0.1))
            last_frame_time = frame_start

            # --- 1. Lấy dữ liệu ---
            state = GetStatus()
            if not printed_state:
                print("[UCR 2026] GetStatus() =", state)
                printed_state = True
            real_speed = read_speed(state, current_speed)
            raw_image = GetRaw()
            if raw_image is None:
                AVControl(0.0, 0.0)
                time.sleep(0.005)
                continue
            height, width = raw_image.shape[:2]

            # --- 2. Thị giác ---
            lane_mask, roi_image = get_lane_mask(raw_image)
            y_offset = height - roi_image.shape[0]

            # --- 3. Tâm làn, 5 điểm xem trước ---
            result = lane_detector.compute(lane_mask)

            # Ngã ba / ngã tư: đặt TRƯỚC LineGuard và PreviewSteering.
            # Dùng 'steering' của khung TRƯỚC (chưa bị ghi đè trong khung này) làm tín hiệu
            # "có đang cua gắt thật hay không" để lọc bớt báo nhầm ở control_module.py.
            is_inter = inter_guard.update(result.seen, steering, result.dev)

            # --- 3a. Điều khiển lái + tốc độ ---
            # Máy trạng thái ngã ba/tư: lái về thẳng + đi thẳng; nếu xe lắc thì dừng chỉnh lái rồi đi tiếp.
            drive = inter_drive.update(
                is_inter, steering, steer_ctl, current_speed, real_speed, dt,
                lane=(result.dev, result.seen))
            in_inter = drive is not None
            if in_inter:
                steering, current_speed = drive
            elif result.valid and any(result.seen[:2]):
                # Chỉ tin bộ lái đầy đủ khi có ÍT NHẤT 1 trong 2 điểm GẦN thấy vạch. Nếu near mất
                # hết mà chỉ có điểm XA thấy (thường là nhiễu ở ngã ba/tư nhỏ), lane_logic.py sẽ
                # fallback e_near sang điểm xa đó -> tâm sai, xe quẹo sai. Trường hợp này rơi vào
                # nhánh else bên dưới (giảm lái về thẳng dần) thay vì tin theo tâm sai.
                steering = steer_ctl.compute(result.e_near, result.e_far, real_speed, dt)
                current_speed = compute_speed_command(
                    steering, result.dev, result.seen, result.vis, current_speed, real_speed, dt,
                    MAX_SPEED, CRUISE_SPEED, MIN_SPEED)
            else:
                steering = steer_ctl.decay(0.80)
                current_speed = max(current_speed - 30.0 * dt, MIN_SPEED)

            # --- 3b. Sắp cán vạch liền -> DỪNG + đánh lái tránh vạch ---
            raw_guard = result.guard if result.guard in GUARD_SIDES else None
            if abs(steering) > GUARD_MAX_STEER:          # đang vào cua -> bỏ qua guard
                raw_guard = None
            guard_cmd = None
            if in_inter:
                line_guard.reset()        # ở ngã ba/tư không tin tín hiệu vạch liền
            else:
                side = {"right": 1, "left": -1}.get(raw_guard)
                guard_cmd = line_guard.update(raw_guard is not None, side, dt, result.seen)
            if guard_cmd is not None:
                g_speed, g_steer = guard_cmd
                steering = float(np.clip(steering + g_steer, -1.0, 1.0))
                current_speed = g_speed

            # --- 4. Gửi lệnh & debug ---
            AVControl(current_speed, steer_to_angle(steering))

            if DEBUG_PRINT_EVERY and frame_start - last_print >= DEBUG_PRINT_EVERY:
                last_print = frame_start
                print(f"cmd={current_speed:5.1f} real={real_speed:5.1f} steer={steering:+.2f} "
                      f"ang={steer_to_angle(steering):+5.1f} near={result.e_near:+.2f} far={result.e_far:+.2f} "
                      f"seen={''.join('1' if x else '0' for x in result.seen)} "
                      f"guard={'STOP' if guard_cmd is not None else (raw_guard or '-')} "
                      f"inter={inter_drive.name if in_inter else '-'} psi={inter_drive.est_psi:+.2f} e={inter_drive.est_e:+.2f}")

            dbg = raw_image.copy()
            cv2.line(dbg, (int(result.xc), 0), (int(result.xc), height), (0, 255, 0), 2)   # trục xe
            for lx, rx, ly in result.lines:                                                      # vạch trái (xanh dương) / phải (tím)
                if lx is not None:
                    cv2.circle(dbg, (int(lx), int(ly) + y_offset), 4, (255, 128, 0), -1)
                if rx is not None:
                    cv2.circle(dbg, (int(rx), int(ly) + y_offset), 4, (255, 0, 255), -1)
            if result.valid and len(result.centers) > 1:
                pl = np.array([(int(cx), int(cy) + y_offset) for cx, cy in result.centers], np.int32)
                cv2.polylines(dbg, [pl], False, (0, 0, 255), 2)
            for i, (px, py, pu, ok) in enumerate(result.points):        # 5 điểm xem trước
                if not ok:
                    col = (150, 150, 150)                                # không thấy
                elif abs(result.dev[i]) > 0.35:
                    col = (0, 0, 255)                                    # lệch nhiều
                elif abs(result.dev[i]) > 0.12:
                    col = (0, 165, 255)                                  # bắt đầu lệch -> phanh
                else:
                    col = (0, 255, 255)                                  # thẳng
                cv2.circle(dbg, (int(px), int(py) + y_offset), 7, col, -1)
                cv2.putText(dbg, str(i + 1), (int(px) + 8, int(py) + y_offset - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            cv2.putText(dbg, f"Cmd {current_speed:.0f} Real {real_speed:.0f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(dbg, f"Steer {steering:+.2f} near {result.e_near:+.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(dbg, "dev " + " ".join(f"{d:+.2f}" for d in result.dev), (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if in_inter and inter_drive.state == IntersectionDrive.FIX:
                cv2.putText(dbg, "NGA BA/TU: XE LAC -> DUNG CHINH LAI", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif in_inter:
                cv2.putText(dbg, "NGA BA/NGA TU -> LAI THANG, DI THANG", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            elif guard_cmd is not None:
                cv2.putText(dbg, f"STOP: vach lien {raw_guard}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif raw_guard:      # đang chờ xác nhận (chưa đủ số khung liên tiếp)
                cv2.putText(dbg, f"canh bao vach {raw_guard} ({line_guard._active_streak}/{GUARD_CONFIRM_FRAMES})",
                            (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

            cv2.imshow("1. UCR 2026 - Front Camera", dbg)
            # Mask hiển thị: vạch tô xanh lá, 5 hàng quét kẻ vạch đỏ mờ -> dễ soi bản đồ mới
            mask_view = cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR)
            mask_view[lane_mask > 0] = (0, 255, 0)
            mh = lane_mask.shape[0]
            for r in lane_detector.rows:
                cv2.line(mask_view, (0, int(r * (mh - 1))), (lane_mask.shape[1], int(r * (mh - 1))), (0, 0, 160), 1)
            cv2.line(mask_view, (int(result.xc), 0), (int(result.xc), mh), (255, 255, 0), 1)
            cv2.imshow("2. UCR 2026 - Lane Mask (Pro)", mask_view)
            cv2.imshow("3. UCR 2026 - Cuted ROI", roi_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            elapsed = time.perf_counter() - frame_start
            if FRAME_TIME - elapsed > 0:
                time.sleep(FRAME_TIME - elapsed)

    except KeyboardInterrupt:
        print("\n[UCR 2026] Dừng chương trình bởi người dùng.")
    finally:
        AVControl(0.0, 0.0)
        CloseSocket()
        cv2.destroyAllWindows()
        print("[UCR 2026] Đã đóng kết nối an toàn.")


if __name__ == "__main__":
    main()