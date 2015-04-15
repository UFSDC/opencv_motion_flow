#!/usr/bin/env python

import numpy as np
import cv2
import video
from datetime import datetime

help_message = '''
USAGE: opt_flow.py [<video_source>]

Keys:
 1 - toggle HSV flow visualization
Hue:  
    Min Up = 'r'
    Min Down = 'R'
    Max Down = 't'
    Max Up = 'T'

Saturation:  
    Min Up = 'f'
    Min Down = 'F'
    Max Down = 'g'
    Max Up = 'G'
Value: 
    Min Up = 'v'
    Min Down = 'V'
    Max Down = 'b'
    Max Up = 'B'

'''

class MotionFlow:

    def __init__(self):
        self.show_hsv = False
        self.start_tracking = False

        self.MIN_VAL = 0;
        self.MAX_VAL = 255;
        self.INC_VAL = 10;
        self.HUE_MAX = 179;

        '''
        hueMin          = MIN_VAL;
        hueMax          = HUE_MAX;
        saturationMin   = MIN_VAL;
        saturationMax   = MAX_VAL;
        valueMin        = MIN_VAL;
        valueMax        = MAX_VAL;
        '''

        self.hueMin         = 50;
        self.hueMax         = 70;
        self.saturationMin  = 30;
        self.saturationMax  = 255;
        self.valueMin       = 30;
        self.valueMax       = 150;

        original = None
        startTrackingImage = None

    def draw_hsv(self, flow):

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

    # Write this image to a file with a timestamp
    def save_image(self, image):

        # Create a filename based on the current time
        now = datetime.now()
        filename = str(now.hour) + '-' + str(now.minute)+ '-' + str(now.second) + '.jpg'

        # Save this image to disk
        cv2.imwrite(filename, image)

    def hue_threshold(self, hue):
        cv2.threshold(hue, self.hueMin, self.MAX_VAL, cv2.THRESH_TOZERO, hue)
        cv2.threshold(hue, self.hueMax, self.MAX_VAL, cv2.THRESH_TOZERO_INV, hue)
        cv2.threshold(hue, 1, self.MAX_VAL, cv2.THRESH_BINARY, hue)
        return hue

    def saturation_threshold(self, saturation):
        cv2.threshold(saturation, self.saturationMin, self.MAX_VAL, cv2.THRESH_TOZERO, saturation)
        cv2.threshold(saturation, self.saturationMax, self.MAX_VAL, cv2.THRESH_TOZERO_INV, saturation)
        cv2.threshold(saturation, 1, self.MAX_VAL, cv2.THRESH_BINARY, saturation)
        return saturation

    def value_threshold(self, value):
        cv2.threshold(value, self.valueMin, self.MAX_VAL, cv2.THRESH_TOZERO, value)
        cv2.threshold(value, self.valueMax, self.MAX_VAL, cv2.THRESH_TOZERO_INV, value)
        cv2.threshold(value, 1, self.MAX_VAL, cv2.THRESH_BINARY, value)
        return value

    def handle_input(self, ch):
        if ch == 27: # Escape key
            return False
        if ch == 32: # Spacebar key
            if(self.start_tracking):
                return False
            else:
                self.start_tracking = not self.start_tracking
                self.startTrackingImage = self.original

        # Hue Threshold Controls
        if ch == ord('r'):
            self.hueMin = min(self.HUE_MAX, self.hueMin + self.INC_VAL)
            print 'hueMin is' + str(self.hueMin) 
        if ch == ord('R'):
            self.hueMin = max(self.MIN_VAL, self.hueMin - self.INC_VAL)
            print 'hueMin is' + str(self.hueMin) 
        if ch == ord('t'):
            self.hueMax = max(self.MIN_VAL, self.hueMax - self.INC_VAL)
            print 'hueMax is' + str(self.hueMax) 
        if ch == ord('T'):
            self.hueMax = min(self.HUE_MAX, self.hueMax + self.INC_VAL)
            print 'hueMax is' + str(self.hueMax) 

        # Saturation Threshold Controls
        if ch == ord('f'):
            self.saturationMin = min(self.MAX_VAL, self.saturationMin + self.INC_VAL)
            print 'saturationMin is' + str(self.saturationMin) 
        if ch == ord('F'):
            self.saturationMin = max(self.MIN_VAL, self.saturationMin - self.INC_VAL)
            print 'saturationMin is' + str(self.saturationMin) 
        if ch == ord('g'):
            self.saturationMax = max(self.MIN_VAL, self.saturationMax - self.INC_VAL)
            print 'saturationMax is' + str(self.saturationMax) 
        if ch == ord('G'):
            self.saturationMax = min(self.MAX_VAL, self.saturationMax + self.INC_VAL)
            print 'saturationMax is' + str(self.saturationMax) 

        # Value Threshold Controls
        if ch == ord('v'):
            self.valueMin = min(self.MAX_VAL, self.valueMin + self.INC_VAL)
            print 'valueMin is' + str(self.valueMin) 
        if ch == ord('V'):
            self.valueMin = max(self.MIN_VAL, self.valueMin - self.INC_VAL)
            print 'valueMin is' + str(self.valueMin) 
        if ch == ord('b'):
            self.valueMax = max(self.MIN_VAL, self.valueMax - self.INC_VAL)
            print 'valueMax is' + str(self.valueMax) 
        if ch == ord('B'):
            self.valueMax = min(self.MAX_VAL, self.valueMax + self.INC_VAL)
            print 'valueMax is' + str(self.valueMax) 

        return True


if __name__ == '__main__':
    import sys
    print help_message

    # Read in a CLI telling the program which camera to use
    # Defaults to 0 (webcam)
    try:
        fn = sys.argv[1]
    except:
        fn = 0

    motion_flow = MotionFlow()

    # Setup the Video Capture
    cam = video.create_capture(fn)

    # Read in an image of what the camera sees
    ret, prev = cam.read()

    # Convert the image to grayscale (necessary for optical flow analysis)
    # This is a 3 channel (BGR) to 1 channel (0 - 255) conversion
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    height, width, _ = prev.shape
    compound_image = np.zeros((height, width), np.uint8)

    while True:
        # Read in camera image
        ret, original = cam.read()

        motion_flow.original = original

        # Give it a nice blur and convert it to HSV colorspace
        img = cv2.GaussianBlur(original, (7, 7), 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Split the 3D image into 3 1D images
        hue, saturation, value = cv2.split(img)

        # -- Apply the threshold operations ---
        # Goal is to make the image binary (black + white), setting all values
        # not inside [MIN_RANGE, MAX_RANGE] to black
        hue = motion_flow.hue_threshold(hue)
        saturation = motion_flow.saturation_threshold(saturation)
        value = motion_flow.value_threshold(value)

        # Combine the "active" (white) parts of the HSV channels to compose 1 image
        gray = cv2.bitwise_and(value, hue)
        gray = cv2.bitwise_and(gray, saturation)

        # Do optical flow analysis
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 0.5, 3, 15, 3, 7, 1.5, 0)

        # Set the current greyscale image to be the previous image for the next optical flow calculation
        prevgray = gray

        # Calculate the hsv image
        hsv_flow = motion_flow.draw_hsv(flow)

        # Calculate the compound image from the hsv_flow
        if (motion_flow.start_tracking):
            compound_image = np.add(hsv_flow[:,:, 0], compound_image)

        # Display the images
        cv2.imshow('greyscale', gray)
        cv2.imshow('flow HSV', hsv_flow)
        cv2.imshow('hue', hue)
        cv2.imshow('saturation', saturation)
        cv2.imshow('value', value)
        cv2.imshow('compound', compound_image)

        # --- Handle Keyboard Input --- 
        ch = 0xFF & cv2.waitKey(5)

        # Break out of this loop if the keyboard input tells us to
        if(not motion_flow.handle_input(ch)):
            break

    # Close all the current windows and then display the computed image
    cv2.destroyAllWindows()

    # Turn the 1D compound image into a 3D image
    compound_3D = cv2.merge([compound_image, compound_image, compound_image])

    # Add the final image to the original tracked image
    final_image = cv2.add(motion_flow.startTrackingImage, compound_3D)
    
    # Save the image to disk
    motion_flow.save_image(final_image)

    cv2.imshow('final image', final_image)

    # Wait for keyboard input before closing the program
    cv2.waitKey(0)
    cv2.destroyAllWindows()
