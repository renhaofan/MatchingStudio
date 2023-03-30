# https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html
import numpy as np
import cv2
import argparse

def argconfig():
    parser = argparse.ArgumentParser(description='SIFT Demo.')
    parser.add_argument('input', type=str, default='',
        help='Image path.')
    parser.add_argument('--rich_vis', action='store_true',
        help='Save output frames to a directory (default: False)')


    return parser.parse_args()

if __name__ == '__main__':
    args = argconfig()
    img = cv2.imread(args.input)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create()
    kp = sift.detect(gray, None)

    if not args.rich_vis:
        ret_img = cv2.drawKeypoints(gray, kp, img)
        cv2.imshow('SIFT', ret_img)
    else:
        ret_img = cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('SIFT', ret_img)    

    cv2.waitKey()
    cv2.destroyAllWindows()