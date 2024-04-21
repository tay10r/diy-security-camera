#!venv/bin/python3

from main import run, VideoInput, Config, NotifyService, open_config
import cv2
from cv2.typing import MatLike
from sys import argv
from loguru import logger
from typing import Union, Sequence, Any

class FakeVideoInput(VideoInput):
    def __init__(self, video_path: str):
        self.__video = cv2.VideoCapture(video_path)

    def is_open(self) -> bool:
        return self.__video.isOpened()

    def read_frame(self) -> Union[MatLike, None]:
        success, img = self.__video.read()
        if success:
            return img
        else:
            return None

class FakeNotifyService(NotifyService):
    def notify(self, img: MatLike, people: Sequence[Any], msg: str):
        for person in people:
            cv2.rectangle(img, (person[0], person[1]), (person[0] + person[2], person[1] + person[3]), (0, 255, 0), 1)
        logger.info(msg)
        cv2.imshow('Notify Output', img)
        cv2.waitKey(0)

def test():
    cv2.startWindowThread()
    cfg = open_config('person_detector.json')
    video_input = FakeVideoInput(argv[1])
    notify_service = FakeNotifyService()
    if video_input.is_open():
        run(cfg, video_input, notify_service)
    else:
        logger.error('Failed to open video.')

test()
