#!venv/bin/python3

import requests
from dataclasses import dataclass
from abc import ABC, abstractmethod
from cv2 import Mat, VideoCapture, HOGDescriptor, HOGDescriptor_getDefaultPeopleDetector
from cv2.typing import MatLike
from loguru import logger
from time import time
from typing import Union, Sequence, Any
import json

class VideoInput(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def is_open(self) -> bool:
        return False

    def read_frame(self) -> Union[MatLike, None]:
        return Mat([])

class LiveVideoInput(VideoInput):

    def __init__(self, device_index):
        self.__device = VideoCapture(index=device_index)

    def is_open(self) -> bool:
        return self.__device.isOpened()

    def read_frame(self) -> Union[MatLike, None]:
        success, img = self.__device.read()
        if not success:
            return None
        else:
            return img

class NotifyService(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def notify(self, img: MatLike, people: Sequence[Any], msg: str):
        pass

class NtfyService(NotifyService):
    def __init__(self, hostname: str, topic: str):
        self.__hostname = hostname
        self.__topic = topic

    def notify(self, img: MatLike, people: Sequence[Any], msg: str):
        url = f'https://{self.__hostname}/{self.__topic}'
        requests.post(url, data=msg.encode('utf-8'))

@dataclass
class Config:
    camera_index: int = 0
    ntfy_hostname: str = 'ntfy.sh'
    ntfy_topic: str = ''
    active_timeout: float = 15.0
    detection_threshold: int = 10
    site_name: str = ''

def open_config(config_path: str) -> Config:
    config_dict = {}
    with open(config_path, 'r') as config_file:
        config_dict = json.load(config_file)
    return Config(**config_dict)

def run(config: Config, video_input: VideoInput, notify_service: NotifyService):

    if config.ntfy_topic == '':
        logger.error('Missing topic to publish to.')
        return

    detector = HOGDescriptor()
    detector.setSVMDetector(HOGDescriptor_getDefaultPeopleDetector())
    last_detection_time = None
    is_active = False

    if not video_input.is_open():
        logger.error('Failed to open video input.')
        return

    logger.info('Monitor started.')

    while video_input.is_open():

        img = video_input.read_frame()

        if img is None:
            logger.warning('Failed to read frame.')
            continue

        regions, _ = detector.detectMultiScale(img, winStride=(4, 4), padding=(4, 4))

        if len(regions) > 0:
            logger.info(f'{len(regions)} detections made.')
            last_detection_time = time()
            if not is_active:
                if config.site_name != '':
                    notify_service.notify(img, regions, f'Presence detected in "{config.site_name}".')
                else:
                    notify_service.notify(img, regions, 'Presence detected.')
                is_active = True
        elif is_active:
            if last_detection_time is None:
                is_active = False
            else:
                elapsed = time() - last_detection_time
                if elapsed > config.active_timeout:
                    is_active = False

if __name__ == '__main__':
    config = open_config('config.json')
    run(config, LiveVideoInput(config.camera_index), NtfyService(config.ntfy_hostname, config.ntfy_topic))
