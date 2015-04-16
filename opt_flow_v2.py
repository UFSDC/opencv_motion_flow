#!/usr/bin/env python
"""
USAGE: opt_flow.py [<video_source>]

Keys:
  1   - toggle HSV flow visualization

  ?/? - decrease/increase value of:
                   MIN MAX
        ---------- --- ---
        Hue        R/r t/T
        Saturation F/f g/G
        Value      V/v b/B
"""
from datetime import datetime
import sys

import numpy as np
import cv2

FARNEBACK_PARAMS = [0.5, 3, 15, 3, 7, 1.5, 0]
GAUSSIANBLUR_PARAMS = [(7, 7), 0]


def main():
    print __doc__

    try:
        video_device_number = int(sys.argv[1])
    except IndexError:
        video_device_number = 0
    capture = cv2.VideoCapture(video_device_number)

    hue_bounds = Bounds('hue', lower=50, upper=70, max=179)
    sat_bounds = Bounds('sat', lower=30, upper=255)
    val_bounds = Bounds('val', lower=30, upper=150)
    bounds_functions = {
        'r': hue_bounds.increase_lower,
        'R': hue_bounds.decrease_lower,
        'T': hue_bounds.increase_upper,
        't': hue_bounds.decrease_upper,
        'f': sat_bounds.increase_lower,
        'F': sat_bounds.decrease_lower,
        'G': sat_bounds.increase_upper,
        'g': sat_bounds.decrease_upper,
        'v': val_bounds.increase_lower,
        'V': val_bounds.decrease_lower,
        'B': val_bounds.increase_upper,
        'b': val_bounds.decrease_upper,
    }

    prevgray = get_grayscale(capture)
    height, width = prevgray.shape
    compound_image = np.zeros((height, width), np.uint8)
    compounding = False

    while True:
        _, image = capture.read()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blurred = cv2.GaussianBlur(hsv, *GAUSSIANBLUR_PARAMS)
        hue, saturation, value = cv2.split(blurred)
        apply_threshold(hue, *hue_bounds)
        apply_threshold(saturation, *sat_bounds)
        apply_threshold(value, *val_bounds)
        gray = reduce(cv2.bitwise_and, [value, hue, saturation])
        hsv_flow = calculate_flow(prevgray, gray)

        if compounding:
            compound_image = np.add(hsv_flow[:,:,0], compound_image)

        cv2.imshow('grayscale', gray)
        cv2.imshow('flow HSV', hsv_flow)
        cv2.imshow('hue', hue)
        cv2.imshow('saturation', saturation)
        cv2.imshow('value', value)
        cv2.imshow('compound', compound_image)

        key = wait_and_maybe_get_keypress(20)
        if key == 'escape':
            return
        elif key == 'space':
            if compounding:
                break
            else:
                compounding = True
                start_tracking_image = image
        else:
            func = bounds_functions.get(key)
            if func:
                func()

    # Turn the 1D compound image into a 3D image
    compound_3D = cv2.merge([compound_image]*3)
    final_image = cv2.add(start_tracking_image, compound_3D)
    save_image(final_image)
    cv2.destroyAllWindows()
    show_until_keypress('final_image', final_image)


class Bounds(object):
    def __init__(self, name, lower=None, upper=None, min=0, max=255, increment=10):
        self.name = name
        self.lower = min if lower is None else lower
        self.upper = max if upper is None else upper
        self.min = min
        self.max = max
        self.increment = increment
        assert self.min <= self.lower <= self.upper <= self.max

    def decrease_lower(self):
        self.lower = max(self.lower - self.increment, self.min)
        print self

    def increase_lower(self):
        self.lower = min(self.lower + self.increment, self.upper)
        print self

    def decrease_upper(self):
        self.upper = max(self.upper - self.increment, self.lower)
        print self

    def increase_upper(self):
        self.upper = min(self.upper + self.increment, self.max)
        print self

    def __iter__(self):
        return iter((self.lower, self.upper, self.max))

    def __repr__(self):
        return '<{s.name}: lower={s.lower} upper={s.upper}>'.format(s=self)


def apply_threshold(values, lower, upper, max):
    cv2.threshold(values, lower, max, cv2.THRESH_TOZERO, values)
    cv2.threshold(values, upper, max, cv2.THRESH_TOZERO_INV, values)
    cv2.threshold(values, 1, max, cv2.THRESH_BINARY, values)


def calculate_flow(prevgray, gray):
    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, *FARNEBACK_PARAMS)
    height, width, _ = flow.shape
    # Get both (x and y) pixel correlations between previous and next images
    fx, fy = flow[:,:,0], flow[:,:,1]
    movement_intensity = np.sqrt(fx*fx + fy*fy)
    # Set up 3 dimensional 8-bit matrix for HSV color values
    bgr = np.zeros((height, width, 1), np.uint8)
    # Correlate the black <-> white value with the pixel movement intensity
    bgr[...,0] = np.minimum(movement_intensity*4, 255)
    return bgr


def decipher_key(key):
    key = key & 0xFF
    if key == 27:
        return 'escape'
    elif key == 32:
        return 'space'
    else:
        return chr(key)


def get_grayscale(capture):
    _, image = capture.read()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def save_image(image):
    """Write image to a timestamped file"""
    now = datetime.now()
    filename = '%s-%s-%s.jpg' % (now.hour, now.minute, now.second)
    cv2.imwrite(filename, image)


def show_until_keypress(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def wait_and_maybe_get_keypress(milliseconds):
    key = cv2.waitKey(milliseconds)
    return None if key == -1 else decipher_key(key)


if __name__ == '__main__':
    main()
