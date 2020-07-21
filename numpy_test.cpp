#include <boost/python.hpp>
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include <boost/python/numpy.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/range/algorithm_ext/push_back.hpp>
#include <boost/range/irange.hpp>
#include <iostream>
#include <iterator>
#include <set>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include<boost/date_time/posix_time/posix_time.hpp>

using namespace cv;

namespace fs = boost::filesystem;
namespace p = boost::python;
namespace np = boost::python::numpy;

int main()
{
    Py_Initialize();
    np::initialize();

    auto time_now = boost::posix_time::microsec_clock::universal_time();
    int width = 224;
    int height = 224;
    int channel = 3;
    int sample_per_device = 6;
    int replication_factor = 8;
    int gradient_accl = 16;
    int batches_per_step = 8;

    int size = width * height * channel * sample_per_device * replication_factor * gradient_accl * batches_per_step;
    

    uint8_t * arr = new uint8_t[size];
    int first_dim = sample_per_device * replication_factor * gradient_accl * batches_per_step;
    auto shape = p::make_tuple(first_dim,
                                channel,
                                height,
                                width
                               );
    auto stride = p::make_tuple(width * height * channel,
                                width * height,
                                width,
                                1) ;
    np::dtype dt1 = np::dtype::get_builtin<uint8_t>();
    auto mul_data_ex = np::from_data(arr,
                                     dt1,
                                     shape,
                                     stride,
                                     p::object());

    auto shape_l = p::make_tuple( batches_per_step,
                                gradient_accl,
                                replication_factor,
                                sample_per_device,
                                channel,
                                height,
                                width
                               ); 
    mul_data_ex = mul_data_ex.reshape(shape_l);

    auto time_now1 = boost::posix_time::microsec_clock::universal_time();
    auto time_elapse = time_now1 - time_now;
    int ticks = time_elapse.ticks();
	std::cout << ticks << std::endl;



}
