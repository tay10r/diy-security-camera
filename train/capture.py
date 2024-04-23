#!venv/bin/python3

import cv2
import time
import os
from loguru import logger

def main(capture_path: str = 'tmp', capture_interval: float = 1.0, root: str = 'data/train'):

    entry_name = time.strftime("%Y%m%d-%H%M%S")

    capture_path = os.path.join(root, entry_name)

    if not os.path.exists(capture_path):
        os.mkdir(capture_path)
        logger.info(f'Created capture directory "{capture_path}".')

    params = [
        cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
        cv2.CAP_PROP_FRAME_WIDTH, 1920,
        cv2.CAP_PROP_FRAME_HEIGHT, 1080,
    ]

    camera = cv2.VideoCapture('/dev/video0', apiPreference=cv2.CAP_V4L2, params=params)
    if not camera.isOpened():
        logger.error('Failed to open camera.')
        return


    cv2.startWindowThread()

    while camera.isOpened():
        success, frame = camera.read()
        timestamp = time.time_ns()

        if not success:
            logger.warning('Failed to capture camera frame.')
            continue

        cv2.imshow('Camera View', frame)

        path = f'{capture_path}/{timestamp}.png'
        if not cv2.imwrite(path, frame):
            logger.warning('Failed to save camera frame.')
            continue

        logger.info(f'Captured frame "{path}".')

        if cv2.pollKey() & 0xff == 'q':
            break

main()
