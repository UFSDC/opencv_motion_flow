#!/usr/bin/env python

import numpy as np
import cv2
import video

help_message = '''
USAGE: opt_flow.py [<video_source>]

Keys:
 1 - toggle HSV flow visualization

'''

def draw_hsv(flow):

    # Get the height and width from the image
    height, width = flow.shape[:2]

    # Get both (x and y) pixel correlations between previous and next images
    fx, fy = flow[:,:,0], flow[:,:,1]

    # Calculate the intensity of the pixel movement (distance formula)
    v = np.sqrt(fx*fx+fy*fy)

    # Set up 3 dimensional 8-bit matrix for HSV color values
    # HSV stands for (Hue, Saturation, Value)
    bgr = np.zeros((height, width, 1), np.uint8)

    # Correlate the black <-> white value with the pixel movement intensity
    bgr[...,0] = np.minimum(v*4, 255)

    return bgr

if __name__ == '__main__':
    import sys
    print help_message

    # Read in a CLI telling the program which camera to use
    # Defaults to 0 (webcam)
    try:
        fn = sys.argv[1]
    except:
        fn = 0

    show_hsv = False
    start_tracking = False

    # Setup the Video Capture
    cam = video.create_capture(fn)

    # Read in an image of what the camera sees
    ret, prev = cam.read()

    startTrackingImage = None

    # Convert the image to grayscale (necessary for optical flow analysis)
    # This is a 3 channel (BGR) to 1 channel (0 - 255) conversion
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    while True:
        # Read in camera image
        ret, img = cam.read()

        # Convert it to greyscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Do optical flow analysis
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 0.5, 3, 15, 3, 7, 1.5, 0)

        # Set the current greyscale image to be the previous image for the next optical flow calculation
        prevgray = gray

        # Calculate the hsv image
        hsv_flow = draw_hsv(flow)

        # TODO: Calculate the compound image from the hsv_flow

        # Display the images
        cv2.imshow('greyscale', gray)
        cv2.imshow('flow HSV', hsv_flow)

        ch = 0xFF & cv2.waitKey(5)
        if ch == 27: # Escape key
            break
        if ch == ord('1'):
            show_hsv = not show_hsv
            print 'HSV flow visualization is', ['off', 'on'][show_hsv]
        if ch == 32: # Spacebar key
            start_tracking = not start_tracking
            startTrackingImage = img

    # Close all the current windows and then display the computed image
    cv2.destroyAllWindows()
    cv2.imshow('final image', startTrackingImage)

    # TODO: Write this image to a file with a timestamp

    # Wait for key input before closing the program
    cv2.waitKey(0)

    cv2.destroyAllWindows()
