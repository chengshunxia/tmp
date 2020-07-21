#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include<boost/date_time/posix_time/posix_time.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python.hpp>
#include <iostream>

using namespace cv;
using namespace std;
namespace p = boost::python;
namespace np = boost::python::numpy;

int main( int argc, char** argv )
{
    if( argc != 2)
    {
     cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
     return -1;
    }

    Py_Initialize();
    np::initialize();
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
    //std::cout << "Mat: " << std::endl << resized_image;  
    vector<Mat> bgr_planes;

    split(resized_image,bgr_planes);

    //std::cout << std::endl << "Mat0:" << std::endl << bgr_planes[0];  
    //std::cout << std::endl << "Mat1:" << std::endl << bgr_planes[1];  
    //std::cout << std::endl << "Mat2:" << std::endl << bgr_planes[2]; 


    auto time_now2 = boost::posix_time::microsec_clock::universal_time();    

    auto time_elapse2 = time_now2 - time_now;   
    
    int ticks2 = time_elapse2.ticks();  

    assert(bgr_planes[0].depth() == CV_8U);

    int width = 224;
    int height = 224;
    int channel = 3;

    int size = width * height * channel;
    

    uint8_t * arr = new uint8_t[size];
    for (auto c = 0 ; c < channel; c++) {
        uint8_t * addr = arr + c*width * height;
        std::memcpy(addr,bgr_planes[c].data,sizeof (uint8_t) *  width * height);
    }
    auto shape = p::make_tuple( channel,
                                height,
                                width
                               );
    auto stride = p::make_tuple(width * height,
                                width,
                                1) ;
    np::dtype dt1 = np::dtype::get_builtin<uint8_t>();
    
    auto mul_data_ex = np::from_data(arr,
                                     dt1,
                                     shape,
                                     stride,
                                     p::object());
    auto time_now1 = boost::posix_time::microsec_clock::universal_time();    

    auto time_elapse = time_now1 - time_now;   
    
    int ticks = time_elapse.ticks();    

    std::cout << std::endl <<ticks << std::endl;
    //std::cout << std::endl <<ticks2 << std::endl;
	std::cout << "Selective multidimensional array :: "<<std::endl
            << p::extract<char const *>(p::str(mul_data_ex)) << std::endl ;


    return 0;
}
