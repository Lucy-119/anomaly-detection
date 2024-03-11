// Basic Module
#include <iostream>
#include <math.h>
#include <mutex>
#include <unistd.h>
#include <thread>

// OpenCV Module
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// Namespace Declaration
using namespace cv;
using namespace std;

#define PI 3.141592






int main()
{
   for (int i = 22; i <= 1000; ++i) 
   {    
        
        string filename_before = "/home/gangjoe/2023_Team-SURE_Install_Guide_release/OpenCV/OpenCV_Test/AI_Programming/train/before/adnormal/" + to_string(i) + ".mp4";
        VideoCapture videoCapture(filename_before);

        float videoFPS = videoCapture.get(cv::CAP_PROP_FPS);
        int videoWidth = videoCapture.get(cv::CAP_PROP_FRAME_WIDTH);
        int videoHeight = videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT);
        

        VideoWriter videoWriter;
        string filename_after = "/home/gangjoe/2023_Team-SURE_Install_Guide_release/OpenCV/OpenCV_Test/AI_Programming/train/before/adnormal/after_" + to_string(i) + ".mp4";
        videoWriter.open(filename_after, videoCapture.get(cv::CAP_PROP_FOURCC), videoFPS , cv::Size(videoWidth, videoHeight), true);

        VideoWriter videoWriter_g;
        string filename_after_g = "/home/gangjoe/2023_Team-SURE_Install_Guide_release/OpenCV/OpenCV_Test/AI_Programming/train/before/adnormal/g_" + to_string(i) + ".mp4";
        videoWriter_g.open(filename_after_g, videoCapture.get(cv::CAP_PROP_FOURCC), videoFPS , cv::Size(videoWidth, videoHeight), true);

        VideoWriter videoWriter_sharp1;
        string filename_sharp1 = "/home/gangjoe/2023_Team-SURE_Install_Guide_release/OpenCV/OpenCV_Test/AI_Programming/train/before/adnormal/sharp1_" + to_string(i) + ".mp4";
        videoWriter_sharp1.open(filename_sharp1, videoCapture.get(cv::CAP_PROP_FOURCC), videoFPS , cv::Size(videoWidth, videoHeight), true);

        VideoWriter videoWriter_sharp2;
        string filename_sharp2 = "/home/gangjoe/2023_Team-SURE_Install_Guide_release/OpenCV/OpenCV_Test/AI_Programming/train/before/adnormal/shar2_" + to_string(i) + ".mp4";
        videoWriter_sharp2.open(filename_sharp2, videoCapture.get(cv::CAP_PROP_FOURCC), videoFPS , cv::Size(videoWidth, videoHeight), true);

        

        if (!videoCapture.isOpened()) 
        {
            cout << "Error opening video file: " << filename_before << endl;
            return -1;
        }

        cout<<"/home/gangjoe/2023_Team-SURE_Install_Guide_release/OpenCV/OpenCV_Test/AI_Programming/Normal_after/Normal_" + to_string(i) + ".mp4"<<endl;

        while (1) 
        {
            Mat frame;
            videoCapture >> frame;

            Mat da;
            // bilateralFilter(frame, da, -1, 10, 5);
            // medianBlur(frame, da, 3);

            imshow("frame", frame);

            if (frame.empty())
                break;

            Mat buller_5;
            GaussianBlur(frame, buller_5, Size(), 10);

            // videoWriter_g<<buller_5;

            float alpha = 1.f;
            Mat dst_5=(1+alpha)*frame-alpha*buller_5;

            
            
            // imshow("Video before " + to_string(i), frame);

            

            // imshow("Video after " + to_string(i), dst_5);

            videoWriter<<dst_5;


            float smoothing_mask_data[3][3] = {{1 / 16.f, 1 / 8.f, 1 / 16.f},
                                       {1 / 8.f, 1 / 4.f, 1 / 8.f},
                                       {1 / 16.f, 1 / 8.f, 1 / 16.f}};
            Mat smoothing_mask(3, 3, CV_32F, smoothing_mask_data);

            float sharpening_mask1_data[3][3] = {{-1.f, -1.f, -1.f},
                                                {-1.f, 3.f, -1.f},
                                                {-1.f, -1.f, -1.f}};
            Mat sharpening_mask1(3, 3, CV_32F, sharpening_mask1_data);

            float sharpening_mask2_data[3][3] = {{0.f, -1.f, 0.f},
                                                {-1.f, 1.f, -1.f},
                                                {0.f, -1.f, 0.f}};
            Mat sharpening_mask2(3, 3, CV_32F, sharpening_mask2_data);

            Mat sharpening_out1, sharpening_out2;
            filter2D(frame, sharpening_out1, -1, sharpening_mask1);
            filter2D(frame, sharpening_out2, -1, sharpening_mask2);

            // imshow("sharpening1", sharpening_out1);
            // imshow("sharpening2", sharpening_out2);

            videoWriter_sharp1<<sharpening_out1;

            videoWriter_sharp2<<sharpening_out2;

            

            cv::Mat img_yuv;
            cv::cvtColor(frame, img_yuv, cv::COLOR_BGR2YUV);

            // Equalize the Y channel
            std::vector<cv::Mat> channels;
            cv::split(img_yuv, channels);
            cv::normalize(channels[0], channels[0]);
            cv::merge(channels, img_yuv);
            

            MatND histogramB, histogramG, histogramR;
            int channel_B[] = { 0 };  // Blue
            int channel_G[] = { 1 };  // Green
            int channel_R[] = { 2 };  // Red
            float channel_range[2] = { 0.0 , 255.0 };
            const float* channel_ranges[1] = { channel_range };
            int histSize[1] = { 256 };

            // R, G, B별로 각각 히스토그램을 계산한다.
            calcHist(&dst_5, 1, channel_B, Mat(), histogramB, 1, histSize, channel_ranges);
            calcHist(&dst_5, 1, channel_G, Mat(), histogramG, 1, histSize, channel_ranges);
            calcHist(&dst_5, 1, channel_R, Mat(), histogramR, 1, histSize, channel_ranges);

            // Plot the histogram
            int hist_w = 512; int hist_h = 400;
            int bin_w = cvRound((double)hist_w / histSize[0]);

            Mat histImageB(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
            normalize(histogramB, histogramB, 0, histImageB.rows, NORM_MINMAX, -1, Mat());

            Mat histImageG(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
            normalize(histogramG, histogramG, 0, histImageG.rows, NORM_MINMAX, -1, Mat());

            Mat histImageR(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
            normalize(histogramR, histogramR, 0, histImageR.rows, NORM_MINMAX, -1, Mat());

            for (int i = 1; i < histSize[0]; i++)
            {
                line(histImageB, Point(bin_w * (i - 1), hist_h - cvRound(histogramB.at<float>(i - 1))),
                    Point(bin_w * (i), hist_h - cvRound(histogramB.at<float>(i))),
                    Scalar(255, 0, 0), 2, 8, 0);

                line(histImageB, Point(bin_w * (i - 1), hist_h - cvRound(histogramG.at<float>(i - 1))),
                    Point(bin_w * (i), hist_h - cvRound(histogramG.at<float>(i))),
                    Scalar(0, 255, 0), 2, 8, 0);

                line(histImageB, Point(bin_w * (i - 1), hist_h - cvRound(histogramR.at<float>(i - 1))),
                    Point(bin_w * (i), hist_h - cvRound(histogramR.at<float>(i))),
                    Scalar(0, 0, 255), 2, 8, 0);

            }

            imshow("histImageB_"+ to_string(i), histImageB);
            
           

            cv::MatND histogramB_o, histogramG_o, histogramR_o;
            int channel_B_o[] = { 0 };  // Blue
            int channel_G_o[] = { 1 };  // Green
            int channel_R_o[] = { 2 };  // Red
            float channel_range_o[2] = { 0.0 , 255.0 };
            const float* channel_ranges_o[1] = { channel_range_o };
            int histSize_o[1] = { 256 };

            // R, G, B별로 각각 히스토그램을 계산한다.
            cv::calcHist(&frame, 1, channel_B_o, cv::Mat(), histogramB_o, 1, histSize_o, channel_ranges_o);
            cv::calcHist(&frame, 1, channel_G_o, cv::Mat(), histogramG_o, 1, histSize_o, channel_ranges_o);
            cv::calcHist(&frame, 1, channel_R_o, cv::Mat(), histogramR_o, 1, histSize_o, channel_ranges_o);

            // Plot the histogram
            hist_w = 512; 
            hist_h = 400;
            bin_w = cvRound((double)hist_w / histSize_o[0]);

            cv::Mat histImageB_o(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
            cv::normalize(histogramB_o, histogramB_o, 0, histImageB_o.rows, cv::NORM_MINMAX, -1, cv::Mat());

            cv::Mat histImageG_o(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
            cv::normalize(histogramG_o, histogramG_o, 0, histImageG_o.rows, cv::NORM_MINMAX, -1, cv::Mat());

            cv::Mat histImageR_o(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
            cv::normalize(histogramR_o, histogramR_o, 0, histImageR_o.rows, cv::NORM_MINMAX, -1, cv::Mat());

            for (int i = 1; i < histSize_o[0]; i++)
            {
                cv::line(histImageB_o, cv::Point(bin_w * (i - 1), hist_h - cvRound(histogramB_o.at<float>(i - 1))),
                    cv::Point(bin_w * (i), hist_h - cvRound(histogramB_o.at<float>(i))),
                    cv::Scalar(255, 0, 0), 2, 8, 0);

                cv::line(histImageB_o, cv::Point(bin_w * (i - 1), hist_h - cvRound(histogramG_o.at<float>(i - 1))),
                    cv::Point(bin_w * (i), hist_h - cvRound(histogramG_o.at<float>(i))),
                    cv::Scalar(0, 255, 0), 2, 8, 0);

                cv::line(histImageB_o, cv::Point(bin_w * (i - 1), hist_h - cvRound(histogramR_o.at<float>(i - 1))),
                    cv::Point(bin_w * (i), hist_h - cvRound(histogramR_o.at<float>(i))),
                    cv::Scalar(0, 0, 255), 2, 8, 0);
            }

            // imshow("histImageB_o_"+ to_string(i) , histImageB_o);



            equalizeHist(dst_5, dst_5);


            if (waitKey(30) == 27) // ESC key to exit
                break;
        }
        // videoCapture.release();
        // videoWriter.release();
        destroyAllWindows();
    }

    

    return 0;
}