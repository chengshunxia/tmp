#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include<boost/date_time/posix_time/posix_time.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    if( argc != 2)
    {
     cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
     return -1;
    }

    auto time_now = boost::posix_time::microsec_clock::universal_time();    



    Mat image;
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    Mat resized_image;
    cv::resize(image,resized_image,Size(224,224));
    vector<Mat> bgr_planes;

    split(resized_image,bgr_planes);
    auto time_now1 = boost::posix_time::microsec_clock::universal_time();    

    auto time_elapse = time_now1 - time_now;   
    
    int ticks = time_elapse.ticks();    

    std::cout << ticks << std::endl;


    return 0;
}
