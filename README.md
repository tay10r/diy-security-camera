About
=====

This is a KISS-style person detector made with [OpenCV](https://opencv.org/) and [ntfy](https://ntfy.sh/).

It uses OpenCV's builtin HOG + SVM detector for people and publishes a notification on [ntfy](https://ntfy.sh) when
one or more people are detected in the camera stream.

To use, enter the repository directory with a terminal and run the commands:

```
python3 -m venv venv
. venv/bin/activate && pip3 install -r requirements
```

Create a file called `person_detector.json` and fill in the details.

```json
{
  "camera_index": 0,
  "ntfy_hostname": "ntfy.sh"
  "ntfy_topic": "<ADD_YOUR_TOPIC_ID_HERE>",
  "active_timeout": 15.0,
  "site_name": "<SITE_NAME>"
}
```

The camera index refers to what camera to get the imagery from.

The ntfy hostname is what server you're pushing notifications to.
It could be your own or the public one.

The topic is a string that is generated to identify your notifications.
The recommended way to generate this is using the website to automatically generate one.

The active timeout refers to the timer that begins when a person is detected.
Notifications are not sent every time someone appears in the camera (otherwise you'd be getting spammed with
notifications). Instead, a person getting detected puts the program into an "active mode". It will
remain in active mode until this amount of time elapses with no new detections on people.
A notification isn't sent unless a person is detected while the script is not in active mode.

The site name refers to where the camera is pointing at. It is useful when this script is used for many sites
and you may need to differentiate where the person was detected at. An example may be "Front Yard" or "Living Room".

Once the configuration file is created (in the repo root), you can run the script:

```
./main.py
```

## Caution

It's important to know that the ntfy service is an open service and the information sent to it is not protected.
Be sure to keep all PII information out of the notifications and the topic names difficult to guess (randomized strings
are great for that). This is why the script does not include imagery.
