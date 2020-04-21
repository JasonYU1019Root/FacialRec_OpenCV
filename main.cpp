//CopyRight @ Jason YU.

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

//global variables
double fps = 30;//frames per second
string face_cascade_name = "haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
const static Scalar colors[]={//define different colors for different people
    CV_RGB(255,255,0),//yellow for admin
    CV_RGB(255,255,255),//white for ordinary people
    CV_RGB(255,0,0)};//red for person of interest
//double scale = 2;//the scale factor to zoom frames

int main()
{
    //load cascades
    if(!face_cascade.load(face_cascade_name)){cout<<"Error! Face cascade not loaded.\n";return -1;}
    cout<<"Face cascade successfully loaded.\n";

    Mat frame;//frame read from video
    //char fileName[100];

    //import the video file
    //cout<<"Please enter the Video File name with extension: ";
    //cin>>fileName;cout<<"Importing Video File...\n";
    VideoCapture capture("./Original_Video_360p.avi");
    if(!capture.isOpened())//check if the video file was loaded
    {
        cout<<"Error! Video File not loaded.\n";
        return -1;
    }cout<<"Video File successfully imported.\n";

    //obtain video size for recorder
    Size video_size(capture.get(CAP_PROP_FRAME_WIDTH),capture.get(CAP_PROP_FRAME_HEIGHT));

    //initialize video recorder
    VideoWriter vidRec;
    vidRec.open("Processed_Video.avi",VideoWriter::fourcc('M','J','P','G'),fps,video_size);
    if(!vidRec.isOpened())//check if the video recorder was ready
    {
        cout<<"Error! Video Recorder not ready.\n";
        return -1;
    }cout<<"Video Recorder ready.\n";

    //import a gray template
    Mat temp = imread("./Template_001.jpg",0);
    if(temp.empty())//check if the template was loaded
    {
        cout<<"Error! Template not loaded.\n";
        return -1;
    }cout<<"Template successfully imported.\n";
    resize(temp,temp,Size(100,100));
    //imshow("test_temp",temp);
    //waitKey();

    system("pause");//wait to see system status

    //run facial recognition
    cout<<"Facial Recognition running...\n";
    //run facial rec on each frame of the video file
    while(capture.read(frame))
    {
        vector<Rect>faces;//vector to store faces
        //create zoomed out frames to boost detection
        Mat frame_gray;
        //Mat smallImg(cvRound(frame.rows/scale),cvRound(frame.cols/scale),CV_8UC1);

        //convert the frame to gray to apply Haar-like algorithm
        cvtColor(frame,frame_gray,COLOR_BGR2GRAY);
        //resize the frame using bilinear difference
        //resize(frame_gray,smallImg,smallImg.size(),0,0,INTER_LINEAR);
        //equalizeHist the resized frame
        //equalizeHist(smallImg,smallImg);//enhance the frame
        equalizeHist(frame_gray,frame_gray);//enhance the frame

        //detect faces
        //parameters: (image,objects,scaleFactor,minNeighbors,flags,minSize,maxSize)
        //face_cascade.detectMultiScale(smallImg,faces,1.05,6,0,Size(30,30),Size());
        face_cascade.detectMultiScale(frame_gray,faces,1.05,6,0,Size(30,30),Size());

        //display a message when no face detected
        if(faces.size()<=0)
        {
            //set text box parameters
            string text_noFace = "No Face Detected!";
            int font_face_noFace = FONT_HERSHEY_COMPLEX;
            double font_scale_noFace = 2;
            int thickness_noFace = 2;
            int baseline_noFace;
            //obtain the size of the text box
            Size text_noFace_size = getTextSize(text_noFace,font_face_noFace,font_scale_noFace,thickness_noFace,&baseline_noFace);
            //draw text
            Point origin_noFace;
            origin_noFace.x=cvRound(frame.cols*0.5-text_noFace_size.width*0.5);
            origin_noFace.y=cvRound(frame.rows*0.5+text_noFace_size.height*0.5);
            putText(frame,text_noFace,origin_noFace,font_face_noFace,font_scale_noFace,colors[2],thickness_noFace,8,0);
        }

        //highlight the detected faces
        int i=1;
        for(vector<Rect>::const_iterator r=faces.begin();r!=faces.end();r++,i++)
        {
            //link parameters
            Rect face_rect;
            Mat frameROI;
            face_rect.x=r->x;
            face_rect.y=r->y;
            face_rect.width=r->width;
            face_rect.height=r->height;
            frameROI=frame_gray(face_rect);

            //text highlight
            //matchTemplate algorithm integrated
            //resize(temp,temp,Size(cvRound(r->width),cvRound(r->height)));//automatically resize the template to fit the size of the detected faces to improve accuracy
            Mat result;//create a new array to store results
            //int result_cols=frameROI.cols-temp.cols+1;
            //int result_rows=frameROI.rows-temp.rows+1;
            //result.create(result_cols,result_rows,CV_32FC1);
            //apply the algorithm
            //normalize(result,result,0,1,NORM_MINMAX,-1,Mat());//IMPORTANT! otherwise blank
            matchTemplate(frameROI,temp,result,TM_CCOEFF_NORMED);//best match==1;less the value worse the match
            //obtain locations
            double minVal=-1;
            double maxVal;
            Point minLoc;
            Point maxLoc;
            //Point matchLoc;
            minMaxLoc(result,&minVal,&maxVal,&minLoc,&maxLoc,Mat());
            //match via maximum value (depends)
            //matchLoc=maxLoc;
            //calculate similarity
            double similarity=maxVal*100;
            //if(similarity<0)similarity*=-1;//wired!?
            //highlight zone with maximum similarity (text box)
            //set text box parameters
            string text_1 = "Jason YU";
            string text_2 = "UNKNOWN";
            int font_face = FONT_HERSHEY_COMPLEX;
            double font_scale = 1;
            int thickness = 2;
            int baseline;
            //obtain the size of the text boxes
            Size text_1_size = getTextSize(text_1,font_face,font_scale,thickness,&baseline);
            Size text_2_size = getTextSize(text_2,font_face,font_scale,thickness,&baseline);
            //draw text
            /*Point origin_1;
            origin_1.x=cvRound((r->x+r->width*0.5)*scale-text_1_size.width*0.5);
            origin_1.y=cvRound(r->y*scale);
            Point origin_2;
            origin_2.x=cvRound((r->x+r->width*0.5)*scale-text_2_size.width*0.5);
            origin_2.y=cvRound(r->y*scale);*/
            Point origin_1;
            origin_1.x=cvRound((r->x+r->width*0.5)-text_1_size.width*0.5);
            origin_1.y=cvRound(r->y);
            Point origin_2;
            origin_2.x=cvRound((r->x+r->width*0.5)-text_2_size.width*0.5);
            origin_2.y=cvRound(r->y);
            Scalar color;
            double pos=capture.get(CAP_PROP_POS_MSEC);
            /*if(pos<=10000)//no need to do identification for the first 10s
                {putText(frame,text_1,origin_1,font_face,font_scale,colors[0],thickness,8,0);color=colors[0];}
            else */if(similarity>15)//less the maxVal worse the match
                {putText(frame,text_1,origin_1,font_face,font_scale,colors[0],thickness,8,0);color=colors[0];}
            else
                {putText(frame,text_2,origin_2,font_face,font_scale,colors[1],thickness,8,0);color=colors[1];}
            printf("Face[%d] Similarity: %.2f%\n",i,similarity);

            //the frame had been zoomed out, now zoomed back
            /*rectangle(
                      frame,
                      Point(cvRound((r->x)*scale),cvRound((r->y)*scale)),
                      Point(cvRound((r->x+r->width-1)*scale),cvRound((r->y+r->height-1)*scale)),
                      color,3,8,0);*/
            rectangle(
                      frame,
                      Point(cvRound((r->x)),cvRound((r->y))),
                      Point(cvRound((r->x+r->width-1)),cvRound((r->y+r->height-1))),
                      color,3,8,0);
        }system("cls");

        //display the processed frame
        imshow("Facial Rec Running...",frame);
        waitKey(1000/fps);
        vidRec.write(frame);
    }

    //display a message when the video file was done facial recognizing
    cout<<"Facial Recognition completed. The processed video file was generated.\n";
    //release all objects and resources
    capture.release();vidRec.release();
    destroyAllWindows();

    system("pause");
    return 0;
}
