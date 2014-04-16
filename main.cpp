#include "opencv_libs.h"
#include "opencv2/opencv.hpp"
#include "detectObject.h"
#include "ImageUtils.h"
#include "preprocessFace.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;
using namespace cv;

#ifdef _EiC
#define WIN32
#endif

#if !defined VK_ESCAPE
    #define VK_ESCAPE 0x1B      // Escape character (27)
#endif

static CvMemStorage* storage_face = 0; //Memory Storage to Sore faces
static CvHaarClassifierCascade* cascade_face = 0; 
void detect_and_draw( IplImage* image );
//Haar cascade - if your openc CV is installed at location C:/OpenCV2.0/
// Cascade Classifier file, used for Face Detection.
//const char *faceCascadeFilename  = "haarcascade_frontalface_alt.xml";
//const char *faceCascadeFilename2 = "haarcascade_profileface.xml";

const char *faceCascadeFilename  = "D:\\OPENCV\\opencv244\\data\\haarcascades\\haarcascade_frontalface_alt.xml";
const char *faceCascadeFilename2 = "D:\\OPENCV\\opencv244\\data\\haarcascades\\haarcascade_profileface.xml";
//const char *faceCascadeFilename  = "D:\\OPENCV\\opencv244\\data\\lbpcascades\\lbpcascade_frontalface.xml";
//const char *faceCascadeFilename2 = "D:\\OPENCV\\opencv244\\data\\lbpcascades\\lbpcascade_profileface.xml";

//const char *faceCascadeFilename = "haarcascade_frontalface_alt_tree.xml";  // Haar face detector.
//const char *eyeCascadeFilename1 = "haarcascade_lefteye_2splits.xml";   // Best eye detector for open-or-closed eyes.
//const char *eyeCascadeFilename2 = "haarcascade_righteye_2splits.xml";   // Best eye detector for open-or-closed eyes.
//const char *eyeCascadeFilename1 = "haarcascade_mcs_lefteye.xml";       // Good eye detector for open-or-closed eyes.
//const char *eyeCascadeFilename2 = "haarcascade_mcs_righteye.xml";       // Good eye detector for open-or-closed eyes.
//const char *eyeCascadeFilename1 = "haarcascade_eye.xml";               // Basic eye detector for open eyes only.
//const char *eyeCascadeFilename2 = "haarcascade_eye_tree_eyeglasses.xml"; // Basic eye detector for open eyes if they might wear glasses.

const char *windowName = "MultiViewFaceDetection";   // Name shown in the GUI window.

// Get access to the webcam.
void initWebcam(VideoCapture &videoCapture, int cameraNumber)
{
    // Get access to the default camera.
    try {   // Surround the OpenCV call by a try/catch block so we can give a useful error message!
        videoCapture.open(cameraNumber);
    } catch (cv::Exception &e) {}
    if ( !videoCapture.isOpened() ) {
        cerr << "ERROR: Could not access the camera!" << endl;
        exit(1);
    }
    cout << "Loaded camera " << cameraNumber << "." << endl;
}

void initDetectors(CascadeClassifier &faceCascade,CascadeClassifier &faceCascade2)
{
    // Load the Face Detection cascade classifier xml file.
    try {   // Surround the OpenCV call by a try/catch block so we can give a useful error message!
        faceCascade.load(faceCascadeFilename);
    } catch (cv::Exception &e) {}
    if ( faceCascade.empty() ) {
        cerr << "ERROR: Could not load Face Detection cascade classifier [" << faceCascadeFilename << "]!" << endl;
        cerr << "Copy the file from your OpenCV data folder (eg: 'C:\\OpenCV\\data\\lbpcascades') into this WebcamFaceRec folder." << endl;
        exit(1);
    }
    cout << "Loaded the Face Detection cascade classifier [" << faceCascadeFilename << "]." << endl;
    // Load the Face Detection cascade classifier xml file.
    try {   // Surround the OpenCV call by a try/catch block so we can give a useful error message!
        faceCascade2.load(faceCascadeFilename2);
    } catch (cv::Exception &e) {}
    if ( faceCascade2.empty() ) {
        cerr << "ERROR: Could not load Face Detection cascade classifier [" << faceCascadeFilename2 << "]!" << endl;
        cerr << "Copy the file from your OpenCV data folder (eg: 'C:\\OpenCV\\data\\lbpcascades') into this WebcamFaceRec folder." << endl;
        exit(1);
    }
    cout << "Loaded the Face Detection cascade classifier [" << faceCascadeFilename << "]." << endl;
}

void handleMultiViewFaceDetection(VideoCapture &videoCapture, CascadeClassifier &faceCascade, CascadeClassifier &faceCascade2)
{
    Ptr<FaceRecognizer> model;
    vector<Mat> preprocessedFaces;
    vector<int> faceLabels;
    Mat old_prepreprocessedFace;
    double old_time = 0;

    // Run forever, until the user hits Escape to "break" out of this loop.
	vector<FacePositionInfo> fpi;  // Position of detected face.
	while (true) {
        // Grab the next camera frame. Note that you can't modify camera frames.
        Mat cameraFrame;		
		try {
		  //videoCapture >> cameraFrame;
			videoCapture.read(cameraFrame);
		} catch (cv::Exception) {
			cout << "An exception has accurred" << endl;
			exit(1);
		};		
        //videoCapture >> cameraFrame;
		//videoCapture.read(cameraFrame);
        if( cameraFrame.empty() ) {
            cerr << "ERROR: Couldn't grab the next camera frame." << endl;
            exit(1);
        }
		// Get a copy of the camera frame that we can draw onto.
        Mat displayedFrame;
        cameraFrame.copyTo(displayedFrame);
		
		// Find a face and preprocess it to have a standard size and contrast & brightness.
		fpi.clear();	
        int facetype = multiViewFaceDetection(displayedFrame,faceCascade,faceCascade2,fpi);		
		// Draw an anti-aliased rectangle around the detected face.
        if (fpi.size()>= 0) {
			if(facetype == 1 || facetype == 4 || facetype == 5){
				for(int i=0;i<fpi.size();i++){
					//rectangle(displayedFrame, faceRect, CV_RGB(255, 255, 0), 2, CV_AA);
					line(displayedFrame,cvPoint(fpi[i].x1,fpi[i].y1),cvPoint(fpi[i].x2,fpi[i].y2),CV_RGB(255,255,0),2,CV_AA);
					line(displayedFrame,cvPoint(fpi[i].x1,fpi[i].y1),cvPoint(fpi[i].x4,fpi[i].y4),CV_RGB(255,255,0),2,CV_AA);
					line(displayedFrame,cvPoint(fpi[i].x3,fpi[i].y3),cvPoint(fpi[i].x2,fpi[i].y2),CV_RGB(255,255,0),2,CV_AA);
					line(displayedFrame,cvPoint(fpi[i].x3,fpi[i].y3),cvPoint(fpi[i].x4,fpi[i].y4),CV_RGB(255,255,0),2,CV_AA);
				}
			}else if(facetype == 2){
				for(int i=0;i<fpi.size();i++){
					//rectangle(displayedFrame, faceRect, CV_RGB(255, 255, 0), 2, CV_AA);
					float weight = 0.7;
					Point points[5];
					points[0] = cvPoint((1-weight)*fpi[i].x2 + weight*fpi[i].x1,(1-weight)*fpi[i].y2 + weight*fpi[i].y1);
					points[1] = cvPoint(fpi[i].x2,fpi[i].y2);
					points[2] = cvPoint(fpi[i].x3,fpi[i].y3);;
					points[3] = cvPoint((1-weight)*fpi[i].x3 + weight*fpi[i].x4,(1-weight)*fpi[i].y3 + weight*fpi[i].y4);
					points[4] = cvPoint(0.5*fpi[i].x1 + 0.5*fpi[i].x4,0.5*fpi[i].y1 + 0.5*fpi[i].y4);
					
					line(displayedFrame,points[0],points[1],CV_RGB(255,0,0),2,CV_AA);
					line(displayedFrame,points[1],points[2],CV_RGB(255,0,0),2,CV_AA);
					line(displayedFrame,points[2],points[3],CV_RGB(255,0,0),2,CV_AA);
					line(displayedFrame,points[3],points[4],CV_RGB(255,0,0),2,CV_AA);
					line(displayedFrame,points[4],points[0],CV_RGB(255,0,0),2,CV_AA);
				}
			}else if(facetype == 3){
				for(int i=0;i<fpi.size();i++){
					//rectangle(displayedFrame, faceRect, CV_RGB(255, 255, 0), 2, CV_AA);
					float weight = 0.7;
					Point points[5];
					points[0] = cvPoint(fpi[i].x1,fpi[i].y1);
					points[1] = cvPoint(weight*fpi[i].x2 + (1-weight)*fpi[i].x1,weight*fpi[i].y2 + (1-weight)*fpi[i].y1);
					points[2] = cvPoint(0.5*fpi[i].x2 + 0.5*fpi[i].x3,0.5*fpi[i].y2 + 0.5*fpi[i].y3);
					points[3] = cvPoint(weight*fpi[i].x3 + (1-weight)*fpi[i].x4,weight*fpi[i].y3 + (1-weight)*fpi[i].y4);
					points[4] = cvPoint(fpi[i].x4,fpi[i].y4);
					
					line(displayedFrame,points[0],points[1],CV_RGB(0,255,0),2,CV_AA);
					line(displayedFrame,points[1],points[2],CV_RGB(0,255,0),2,CV_AA);
					line(displayedFrame,points[2],points[3],CV_RGB(0,255,0),2,CV_AA);
					line(displayedFrame,points[3],points[4],CV_RGB(0,255,0),2,CV_AA);
					line(displayedFrame,points[4],points[0],CV_RGB(0,255,0),2,CV_AA);
				}
			}
        }
		// Show the camera frame on the screen.        
		imshow(windowName, displayedFrame);

        // IMPORTANT: Wait for atleast 20 milliseconds, so that the image can be displayed on the screen!
        // Also checks if a key was pressed in the GUI window. Note that it should be a "char" to support Linux.
        char keypress = waitKey(20);  // This is needed if you want to see anything!
		if (keypress == VK_ESCAPE) {   // Escape Key
            // Quit the program!
            break;
        }

    }//end while
}

int main(int argc, char* argv[])
{
	CascadeClassifier faceCascade;
	CascadeClassifier faceCascade2;
    //CascadeClassifier eyeCascade1;
    //CascadeClassifier eyeCascade2;
    initDetectors(faceCascade,faceCascade2);

	VideoCapture videoCapture;	
	if(argc == 2){
		printf("Using File %s\n",argv[1]);
		videoCapture.open(argv[1]);
		if(!videoCapture.isOpened()){
			cout << "Failed to open the fild " << argv[1] << endl;
			exit(1);
		}else{
			long totalFrameNumber = videoCapture.get(CV_CAP_PROP_FRAME_COUNT);  
			cout<<"# of Frame = "<<totalFrameNumber<<endl;
			cout<<"# of Rate = " <<videoCapture.get(CV_CAP_PROP_FPS) << endl;
		}
	}else{
		int cameraNumber = 0;
		initWebcam(videoCapture,cameraNumber);		
	}	
	namedWindow(windowName);
	
	handleMultiViewFaceDetection(videoCapture, faceCascade,faceCascade2);
	
	system("pause");
	return 0;
}

////////////////////////////  Function To detect face //////////////////////////

void detect_and_draw( IplImage* img )
{

	double scale = 2;

	// create a gray image for the input image
	IplImage* gray = cvCreateImage( cvSize(img->width,img->height), 8, 1 );
	// Scale down the ie. make it small. This will increase the detection speed
	IplImage* small_img = cvCreateImage( cvSize( cvRound (img->width/scale),cvRound (img->height/scale)),8, 1 );

	int i;

	cvCvtColor( img, gray, CV_BGR2GRAY );

	cvResize( gray, small_img, CV_INTER_LINEAR );

	// Equalise contrast by eqalizing histogram of image
	cvEqualizeHist( small_img, small_img );

	cvClearMemStorage( storage_face);

	if( cascade_face )
	{
		// Detect object defined in Haar cascade. IN our case it is face
		CvSeq* faces = cvHaarDetectObjects( small_img, cascade_face, storage_face,
			1.1, 2, 0/*CV_HAAR_DO_CANNY_PRUNING*/,
			cvSize(30, 30) );

		// Draw a rectagle around all detected face 
		for( i = 0; i < (faces ? faces->total : 0); i++ )
		{
			CvRect r = *(CvRect*)cvGetSeqElem( faces, i );
			cvRectangle( img, cvPoint(r.x*scale,r.y*scale),cvPoint((r.x+r.width)*scale,(r.y+r.height)*scale),CV_RGB(255,0,0),3,8,0 );

		}
	}

	cvShowImage( "result", img );
	cvReleaseImage( &gray );
	cvReleaseImage( &small_img );
}