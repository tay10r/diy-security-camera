#!venv/bin/python3

import cv2
import os
import json
import sys
from loguru import logger

def load_labels(path: str) -> dict:
    labels = {}
    if os.path.exists(path):
        logger.info('Loading existing label set from "{path}".')
        with open(path, 'r') as labels_file:
            labels = json.load(labels_file)
    else:
        logger.info('Starting new label set.')
    return labels

def format_image_title(name: str, labels: dict) -> str:
    if name in labels:
        contains_person = labels[name] == 1
        if contains_person:
            return 'Person'
        else:
            return 'No Person'
    else:
        return f''

def get_image_label_color(name: str, labels: dict) -> tuple:
    if name in labels:
        if labels[name] == 1:
            return (0, 255, 0)
        else:
            return (0, 0, 255)
    else:
        return (255, 0, 0)

class Program:
    def __init__(self, labels_path: str = 'data/labels.json'):
        if not os.path.exists('data'):
            os.mkdir('data')
        self.__labels = load_labels(labels_path)
        self.__frame_index = 0
        self.__frames = []
        self.__should_exit = False
        self.__labels_path = labels_path

    def open_frames(self, root: str):
        frames = os.listdir(root)
        frames.sort()
        self.__frames = frames
        self.__root = root

    def run(self):
        cv2.startWindowThread()
        while not self.__should_exit:
            self.__run_iteration()
        cv2.destroyAllWindows()

    def __run_iteration(self):
        name = self.__frames[self.__frame_index]
        path = os.path.join(self.__root, name)
        frame = cv2.imread(path)
        cv2.putText(frame,
                    format_image_title(name, self.__labels),
                    org=(20, 20),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1.5,
                    color=get_image_label_color(name, self.__labels),
                    thickness=2,
                    lineType=cv2.LINE_AA)
        title = format_image_title(name, self.__labels)
        cv2.imshow('Label Tool', frame)
        self.__process_key(cv2.waitKey(0) & 0xff)

    def __save_labels(self):
        with open(self.__labels_path, 'w') as labels_file:
            json.dump(self.__labels, labels_file)
            logger.info('Labels saved.')

    def __process_key(self, key: int):
        current_name = self.__frames[self.__frame_index]
        if key == ord('q'):
            self.__save_labels()
            self.__should_exit = True
        elif key == ord('a'):
            if self.__frame_index > 0:
                self.__frame_index -= 1
        elif key == ord('d'):
            if (self.__frame_index + 1) < len(self.__frames):
                self.__frame_index += 1
        elif key == ord('n'):
            # Skip until the next unlabeled frame.
            while self.__frame_index < len(self.__frames) and current_name in self.__labels:
                self.__frame_index += 1
        elif key == ord('0'):
            self.__labels[current_name] = 0
        elif key == ord('1'):
            self.__labels[current_name] = 1
        elif key == ord('s'):
            self.__save_labels()

help_str = """
This program will open a directory of images (created by the image capture program) and provide a means of labeling
each image as having a person in it or not. To control the program, use the following keys:

    'a' - Move to the previous image.
    'd' - Move to the next image.
    '0' - Label the current image as not having a person in it.
    '1' - Label the current image as having a person in it.
    'n' - Go to the next unlabeled image in the directory.
    's' - Save the current set of labels.
    'q' - Save the current set of labels and quit.
"""

def main():

    print(help_str)

    args = sys.argv[1:]
    if len(args) != 1:
        logger.error('Expected scene path as argument.')
        return

    program = Program()
    program.open_frames(args[0])
    program.run()

main()
