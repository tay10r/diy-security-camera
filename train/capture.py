#!../venv/bin/python3

import cv2
import time
import os
from loguru import logger

def main(camera_index: int = 0, capture_interval: float = 1.0, capture_dir: str = 'data'):
    camera = cv2.VideoCapture(index=0)
    if not camera.isOpened():
        logger.error('Failed to open camera.')
        return

    while camera.isOpened():
        success, frame = camera.read()
        if not success:
            logger.warning('Failed to capture camera frame.')
            continue

        timestamp = time.time_ns()
        path = f'{capture_dir}/{timestamp}.jpg'
        if not cv2.imwrite(path, frame):
            logger.warning('Failed to save camera frame.')
            continue

        time.sleep(capture_interval)

main()