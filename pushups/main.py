import time
from pathlib import Path

import cv2
import numpy as np
from playsound3 import playsound
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator


DEVICE = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / 'yolo26n-pose.pt'
SOUND_FILE = str(ROOT_DIR / 'acolyteyes2.mp3')


def get_angle(a, b, c):
    cb = np.atan2(c[1] - b[1], c[0] - b[0])
    ab = np.atan2(a[1] - b[1], a[0] - b[0])
    angle = np.rad2deg(cb - ab)
    angle = angle + 360 if angle < 0 else angle
    return 360 - angle if angle > 180 else angle


def detect_pull_up(annotated, keypoints, is_hanging, count):
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_elbow = keypoints[7]
    right_elbow = keypoints[8]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]

    trigger = False

    if left_shoulder[0] > 0 and right_shoulder[0] > 0:
        l_angle = get_angle(left_shoulder, left_elbow, left_wrist)
        r_angle = get_angle(right_shoulder, right_elbow, right_wrist)

        if l_angle > 150 and r_angle > 150 and not is_hanging:
            is_hanging = True

        if l_angle < 100 and r_angle < 100 and is_hanging:
            is_hanging = False
            count += 1
            trigger = True

        cv2.putText(
            annotated,
            f'Pullups: {count}',
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3,
        )

    return is_hanging, count, trigger


def main():
    model = YOLO(str(MODEL_PATH))
    model.to(DEVICE)

    playsound(SOUND_FILE, block=False)

    camera = cv2.VideoCapture(0)
    is_hanging = False
    count = 0
    last_seen_time = time.time()

    while camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            break

        if time.time() - last_seen_time > 5.0:
            count = 0
            is_hanging = False

        cv2.imshow('Camera', frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break

        t = time.perf_counter()
        results = model.predict(frame)
        elapsed = 1 / (time.perf_counter() - t)
        print(f'Elapsed time {elapsed:.1f}')

        if not results:
            continue

        result = results[0]
        keypoints = result.keypoints.xy.tolist()
        if not keypoints or len(keypoints[0]) < 11:
            continue

        print(keypoints)
        last_seen_time = time.time()

        annotator = Annotator(frame)
        annotator.kpts(result.keypoints.data[0], result.orig_shape, 5, True)
        annotated = annotator.result()

        is_hanging, count, trigger = detect_pull_up(
            annotated, keypoints[0], is_hanging, count
        )

        if trigger:
            playsound(SOUND_FILE, block=False)

        cv2.imshow('Pose', annotated)

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
