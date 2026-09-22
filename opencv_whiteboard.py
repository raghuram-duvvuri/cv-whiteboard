import cv2
import numpy as np
import mediapipe as mp
import time
import os

CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 720
BRUSH_MIN, BRUSH_MAX = 2, 30
DEFAULT_BRUSH = 5
SMOOTHING = 0.6
SAVE_PATH = "whiteboard_output.png"

PINCH_MIN_DIST = 20
PINCH_MAX_DIST = 250

PICKER_W = 520
PICKER_H = 250
PICKER_Y = 24
PICKER_SV_X = 18
PICKER_SV_Y = 46
PICKER_SV_W = 300
PICKER_SV_H = 180
PICKER_HUE_X = 335
PICKER_HUE_Y = 46
PICKER_HUE_W = 28
PICKER_HUE_H = 180

GESTURES = [
    ("1 finger","Draw"),
    ("2 fingers","Erase"),
    ("3 fingers","RGB color picker"),
    ("Open palm","Clear canvas"),
    ("Pinch","Brush size"),
    ("Fist","Idle"),
]
LEGEND_W = 270
LEGEND_PADDING = 12
LEGEND_LINE_H = 25

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def fingers_up(hand_landmarks, handedness_label):

    lm = hand_landmarks.landmark
    fingers = []

    if handedness_label == "Right":
        fingers.append(lm[mp_hands.HandLandmark.THUMB_TIP].x < lm[mp_hands.HandLandmark.THUMB_IP].x)
    else:
        fingers.append(lm[mp_hands.HandLandmark.THUMB_TIP].x > lm[mp_hands.HandLandmark.THUMB_IP].x)

    tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_TIP,
            mp_hands.HandLandmark.PINKY_TIP]
    pips = [mp_hands.HandLandmark.INDEX_FINGER_PIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            mp_hands.HandLandmark.RING_FINGER_PIP,
            mp_hands.HandLandmark.PINKY_PIP]

    for tip, pip in zip(tips, pips):
        fingers.append(lm[tip].y < lm[pip].y)

    return fingers


def pinch_distance(hand_landmarks, w, h):
    lm = hand_landmarks.landmark
    tx, ty = lm[mp_hands.HandLandmark.THUMB_TIP].x * w, lm[mp_hands.HandLandmark.THUMB_TIP].y * h
    ix, iy = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w, lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h
    return ((tx - ix) ** 2 + (ty - iy) ** 2) ** 0.5, (int(tx), int(ty)), (int(ix), int(iy))


def hsv_to_bgr(h, s, v):
    """Convert HSV values to an OpenCV BGR color."""
    hsv = np.uint8([[[int(h) % 180, int(s), int(v)]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def draw_color_picker(frame, hue, sat, val, cursor=None):
    
    x0 = (FRAME_W - PICKER_W) // 2
    y0 = PICKER_Y

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + PICKER_W, y0 + PICKER_H), (24, 24, 24), -1)
    cv2.addWeighted(overlay, 0.90, frame, 0.10, 0, frame)

    cv2.putText(frame, "RGB COLOR", (x0 + 14, y0 + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    
    sv_x = x0 + PICKER_SV_X
    sv_y = y0 + PICKER_SV_Y
    hsv = np.zeros((PICKER_SV_H, PICKER_SV_W, 3), dtype=np.uint8)
    for yy in range(PICKER_SV_H):
        v = 255 - int(yy * 255 / max(PICKER_SV_H - 1, 1))
        for xx in range(PICKER_SV_W):
            s = int(xx * 255 / max(PICKER_SV_W - 1, 1))
            hsv[yy, xx] = (int(hue) % 180, s, v)
    sv_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    frame[sv_y:sv_y + PICKER_SV_H, sv_x:sv_x + PICKER_SV_W] = sv_img
    cv2.rectangle(frame, (sv_x, sv_y),
                  (sv_x + PICKER_SV_W, sv_y + PICKER_SV_H), (220, 220, 220), 1)


    hue_x = x0 + PICKER_HUE_X
    hue_y = y0 + PICKER_HUE_Y
    hue_img = np.zeros((PICKER_HUE_H, PICKER_HUE_W, 3), dtype=np.uint8)
    for yy in range(PICKER_HUE_H):
        h = int((1 - yy / max(PICKER_HUE_H - 1, 1)) * 179)
        hue_img[yy, :, :] = hsv_to_bgr(h, 255, 255)
    frame[hue_y:hue_y + PICKER_HUE_H, hue_x:hue_x + PICKER_HUE_W] = hue_img
    cv2.rectangle(frame, (hue_x, hue_y),
                  (hue_x + PICKER_HUE_W, hue_y + PICKER_HUE_H), (220, 220, 220), 1)

    current_bgr = hsv_to_bgr(hue, sat, val)
    preview_x = x0 + 385
    preview_y = y0 + 72
    cv2.rectangle(frame, (preview_x, preview_y), (preview_x + 100, preview_y + 70),
                  current_bgr, -1)
    cv2.rectangle(frame, (preview_x, preview_y), (preview_x + 100, preview_y + 70),
                  (255, 255, 255), 1)

    b, g, r = current_bgr
    cv2.putText(frame, f"R {r:3d}", (preview_x, y0 + 164),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
    cv2.putText(frame, f"G {g:3d}", (preview_x, y0 + 184),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
    cv2.putText(frame, f"B {b:3d}", (preview_x, y0 + 204),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)

    if cursor is not None:
        cx, cy = cursor
        if sv_x <= cx <= sv_x + PICKER_SV_W and sv_y <= cy <= sv_y + PICKER_SV_H:
            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), 2)
        elif hue_x <= cx <= hue_x + PICKER_HUE_W and hue_y <= cy <= hue_y + PICKER_HUE_H:
            cv2.rectangle(frame, (hue_x - 3, cy - 3), (hue_x + hue_x * 0 + PICKER_HUE_W + 3, cy + 3),
                          (255, 255, 255), 2)


def color_picker_hit(x, y, hue, sat, val):
    x0 = (FRAME_W - PICKER_W) // 2
    y0 = PICKER_Y
    sv_x = x0 + PICKER_SV_X
    sv_y = y0 + PICKER_SV_Y
    hue_x = x0 + PICKER_HUE_X
    hue_y = y0 + PICKER_HUE_Y

    if sv_x <= x <= sv_x + PICKER_SV_W and sv_y <= y <= sv_y + PICKER_SV_H:
        sat = int((x - sv_x) * 255 / max(PICKER_SV_W - 1, 1))
        val = 255 - int((y - sv_y) * 255 / max(PICKER_SV_H - 1, 1))
    elif hue_x <= x <= hue_x + PICKER_HUE_W and hue_y <= y <= hue_y + PICKER_HUE_H:
        hue = int((1 - (y - hue_y) / max(PICKER_HUE_H - 1, 1)) * 179)

    return hue, sat, val


def draw_legend(frame, visible):
    if not visible:
        cv2.putText(frame, "Press 'g' for instructions", (FRAME_W - 225, FRAME_H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1)
        return

    x0 = FRAME_W - LEGEND_W - 14
    y0 = 18
    h = 42 + LEGEND_LINE_H * len(GESTURES)
    y1 = y0 + h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + LEGEND_W, y1), (24, 24, 24), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + LEGEND_W, y1), (120, 120, 120), 1)

    cv2.putText(frame, "GESTURES", (x0 + LEGEND_PADDING, y0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
    cv2.putText(frame, "'g' hide", (x0 + LEGEND_W - 64, y0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (170, 170, 170), 1)

    ty = y0 + 43
    for gesture_name, action in GESTURES:
        cv2.putText(frame, gesture_name, (x0 + LEGEND_PADDING, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 255), 1)
        cv2.putText(frame, action, (x0 + 105, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.39, (240, 240, 240), 1)
        ty += LEGEND_LINE_H


def main():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        print("ERROR: could not open webcam. Check CAM_INDEX or camera permissions.")
        return

    canvas = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

    current_color_name = "RGB(255,0,0)"
    current_color = (0, 0, 255)  # OpenCV BGR for RGB(255,0,0)
    brush_size = DEFAULT_BRUSH
    saved_brush_size = DEFAULT_BRUSH

    picker_hue, picker_sat, picker_val = 0, 255, 255  # red

    prev_x, prev_y = None, None
    smoothed_x, smoothed_y = None, None
    legend_visible = True
    clear_gesture_latched = False

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    prev_time = time.time()

    print("Hand-Tracking Whiteboard running. Press 'q' or Esc to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read from webcam.")
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        mode_label = "NO HAND"

        if results.multi_hand_landmarks and results.multi_handedness:
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness_label = results.multi_handedness[0].classification[0].label

            h, w = FRAME_H, FRAME_W
            idx_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(idx_tip.x * w), int(idx_tip.y * h)

            if smoothed_x is None:
                smoothed_x, smoothed_y = x, y
            else:
                smoothed_x = int(SMOOTHING * smoothed_x + (1 - SMOOTHING) * x)
                smoothed_y = int(SMOOTHING * smoothed_y + (1 - SMOOTHING) * y)
            x, y = smoothed_x, smoothed_y

            thumb, index, middle, ring, pinky = fingers_up(hand_landmarks, handedness_label)
            up_count = sum([index, middle, ring, pinky])

            if up_count >= 4:
                mode_label = "CLEAR"
                prev_x, prev_y = None, None
            
                if not clear_gesture_latched:
                    canvas[:] = 0
                    clear_gesture_latched = True

            elif thumb and index and not middle and not ring and not pinky:
                mode_label = "BRUSH SIZE"
                prev_x, prev_y = None, None
                clear_gesture_latched = False
                dist, thumb_pt, index_pt = pinch_distance(hand_landmarks, FRAME_W, FRAME_H)
                dist = max(PINCH_MIN_DIST, min(PINCH_MAX_DIST, dist))
                center_dist = PINCH_MIN_DIST + (
                    (saved_brush_size - BRUSH_MIN) / (BRUSH_MAX - BRUSH_MIN)
                ) * (PINCH_MAX_DIST - PINCH_MIN_DIST)
                delta = dist - center_dist
                brush_size = int(round(saved_brush_size + delta * (
                    (BRUSH_MAX - BRUSH_MIN) / (PINCH_MAX_DIST - PINCH_MIN_DIST)
                )))
                brush_size = max(BRUSH_MIN, min(BRUSH_MAX, brush_size))
                saved_brush_size = brush_size
                cv2.line(frame, thumb_pt, index_pt, (255, 255, 255), 2)
                cv2.circle(frame, thumb_pt, 8, (255, 0, 255), -1)
                cv2.circle(frame, index_pt, 8, (255, 0, 255), -1)
                cv2.putText(frame, f"Brush: {brush_size}", (index_pt[0] + 15, index_pt[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            elif index and middle and ring and not pinky:
                mode_label = "COLOR PICKER"
                prev_x, prev_y = None, None
                clear_gesture_latched = False
                picker_hue, picker_sat, picker_val = color_picker_hit(
                    x, y, picker_hue, picker_sat, picker_val
                )
                current_color = hsv_to_bgr(picker_hue, picker_sat, picker_val)
                b, g, r = current_color
                current_color_name = f"RGB({r},{g},{b})"
                draw_color_picker(
                    frame, picker_hue, picker_sat, picker_val, (x, y)
                )

            elif index and middle and not ring and not pinky:
                mode_label = "ERASE"
                prev_x, prev_y = None, None
                clear_gesture_latched = False
                cv2.circle(frame, (x, y), brush_size, (0, 0, 0), 2)
                if y > 5:
                    cv2.circle(canvas, (x, y), brush_size, (0, 0, 0), -1)

            elif index and not middle and not ring and not pinky:
                mode_label = "DRAW"
                clear_gesture_latched = False
                if y > 5:
                    if prev_x is None:
                        prev_x, prev_y = x, y
                    cv2.line(canvas, (prev_x, prev_y), (x, y), current_color, brush_size)
                    prev_x, prev_y = x, y
                else:
                    prev_x, prev_y = None, None
                cv2.circle(frame, (x, y), brush_size // 2 + 2, current_color, -1)

            else:
                mode_label = "IDLE"
                prev_x, prev_y = None, None
                clear_gesture_latched = False

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            prev_x, prev_y = None, None
            smoothed_x, smoothed_y = None, None
        
        
        mask = canvas.sum(axis=2) > 0
        frame[mask] = canvas[mask]

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(frame, f"Mode: {mode_label}   Color: {current_color_name}   Brush: {brush_size}",
                    (14, FRAME_H - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (255, 255, 255), 2)
        cv2.putText(frame, "[c] Clear   [s] Save   [+/-] Brush size   [g] Instructions   [q]/Esc Quit",
                    (14, FRAME_H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (200, 200, 200), 1)

        draw_legend(frame, legend_visible)

        cv2.imshow("Hand-Tracking Whiteboard", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # q or Esc
            break
        elif key == ord('c'):
            canvas[:] = 0
        elif key == ord('s'):
            cv2.imwrite(SAVE_PATH, canvas)
            print(f"Saved canvas to {os.path.abspath(SAVE_PATH)}")
        elif key in (ord('+'), ord('=')):
            brush_size = min(BRUSH_MAX, brush_size + 2)
            saved_brush_size = brush_size
        elif key in (ord('-'), ord('_')):
            brush_size = max(BRUSH_MIN, brush_size - 2)
            saved_brush_size = brush_size
        elif key == ord('g'):
            legend_visible = not legend_visible

    hands.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
