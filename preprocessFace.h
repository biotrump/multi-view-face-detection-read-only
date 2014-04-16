#pragma once

#include <stdio.h>
#include <iostream>
#include <vector>

// Include OpenCV's C++ Interface
#include "opencv2/opencv.hpp"


using namespace cv;
using namespace std;

struct FacePositionInfo{
    int x1; int y1;
    int x2; int y2;
    int x3; int y3;
    int x4; int y4; 
};

#define M_PI 3.141592654
#define SCALEDWIDTH 240
/*
// Remove the outer border of the face, so it doesn't include the background & hair.
// Keeps the center of the rectangle at the same place, rather than just dividing all values by 'scale'.
Rect scaleRectFromCenter(const Rect wholeFaceRect, float scale = 0.7f);
*/

// Search for both eyes within the given face image. Returns the eye centers in 'leftEye' and 'rightEye',
// or sets them to (-1,-1) if each eye was not found. Note that you can pass a 2nd eyeCascade if you
// want to search eyes using 2 different cascades. For example, you could use a regular eye detector
// as well as an eyeglasses detector, or a left eye detector as well as a right eye detector.
// Or if you don't want a 2nd eye detection, just pass an uninitialized CascadeClassifier.
// Can also store the searched left & right eye regions if desired.
void detectBothEyes(const Mat &face, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2, Point &leftEye, Point &rightEye, Rect *searchedLeftEye = NULL, Rect *searchedRightEye = NULL);

// Histogram Equalizae seperately for the left and right sides of the face,
// so that if there is a strong light on one side but not the other, it will still look OK.
void equalizeLeftAndRightHalves(Mat &faceImg);

// Create a grayscale face image that has a standard size and contrast & brightness.
// "srcImg" should be a copy of the whole color camera frame, so that it can draw the eye positions onto.
// If 'doLeftAndRightSeparately' is true, it will process left & right sides seperately,
// so that if there is a strong light on one side but not the other, it will still look OK.
// Performs Face Preprocessing as a combination of:
//  - geometrical scaling, rotation and translation using Eye Detection,
//  - smoothing away image noise using a Bilateral Filter,
//  - standardize the brightness on both left and right sides of the face independently using separated Histogram Equalization,
//  - removal of background and hair using an Elliptical Mask.
// Returns either a preprocessed face square image or NULL (ie: couldn't detect the face and 2 eyes).
// If a face is found, it can store the rect coordinates into 'storeFaceRect' and 'storeLeftEye' & 'storeRightEye' if given,
// and eye search regions into 'searchedLeftEye' & 'searchedRightEye' if given.
Mat getPreprocessedFace(Mat &srcImg, int desiredFaceWidth, CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2, bool doLeftAndRightSeparately, Rect *storeFaceRect = NULL, Point *storeLeftEye = NULL, Point *storeRightEye = NULL, Rect *searchedLeftEye = NULL, Rect *searchedRightEye = NULL);
int multiViewFaceDetection(Mat &srcImg, CascadeClassifier &faceCascade, CascadeClassifier &faceCascade2, vector<FacePositionInfo>& fpi);
void GS_rotate( IplImage* srcImg, IplImage *srcImgRotated, int angle);
void rect2FPI(Rect& faceRect, FacePositionInfo& fpi, int angle, int width, int height);


