#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include<boost/date_time/posix_time/posix_time.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python.hpp>

#include <boost/thread/thread.hpp>
#include <iostream>
#include <fstream>

#include "blocking_queue.hpp"
#include <boost/chrono.hpp>
#include <boost/thread/thread.hpp> 
#include <boost/thread/latch.hpp>



using namespace cv;
using namespace std;
namespace p = boost::python;
namespace np = boost::python::numpy;


#include <boost/thread.hpp>
#include <string>
#include <iostream>

// #include "caffe/layers/base_data_layer.hpp"
// #include "caffe/parallel.hpp"
// #include "caffe/util/blocking_queue.hpp"

#include "blocking_queue.hpp"

template<typename T>
class BlockingQueue<T>::sync {
 public:
  mutable boost::mutex mutex_;
  boost::condition_variable condition_;
};
 
template<typename  T>
BlockingQueue<T>::BlockingQueue()
    : sync_(new sync()) {
}

template<typename T>
void BlockingQueue<T>::push(const T& t) {
  boost::mutex::scoped_lock lock(sync_->mutex_);
  queue_.push(t);
  lock.unlock();
  sync_->condition_.notify_one();
}

template<typename T>
bool BlockingQueue<T>::try_pop(T* t) {
  boost::mutex::scoped_lock lock(sync_->mutex_);

  if (queue_.empty()) {
    return false;
  }
  *t = queue_.front();
  queue_.pop();
  return true;
}

template<typename T>
T BlockingQueue<T>::pop() {
  boost::mutex::scoped_lock lock(sync_->mutex_);

  while (queue_.empty()) {
    sync_->condition_.wait(lock);
  }

  T t = queue_.front();
  queue_.pop();
  return t;
}

template<typename T>
bool BlockingQueue<T>::try_peek(T* t) {
  boost::mutex::scoped_lock lock(sync_->mutex_);

  if (queue_.empty()) {
    return false;
  }
  *t = queue_.front();
  return true;
}

template<typename T>
T BlockingQueue<T>::peek() {
  boost::mutex::scoped_lock lock(sync_->mutex_);

  while (queue_.empty()) {
    sync_->condition_.wait(lock);
  }

  return queue_.front();
}

template<typename T>
size_t BlockingQueue<T>::size() const {
  boost::mutex::scoped_lock lock(sync_->mutex_);
  return queue_.size();
}


struct imgFileReadRequest {
    std::string imgFilePath;
    float *arr; 
    int index;
    boost::latch* gen_latch;
};

BlockingQueue<imgFileReadRequest*> imgReadQueue;

void readimg_thread() {
    while (1) {
        auto request = imgReadQueue.pop();
        auto filename = request->imgFilePath;
        float * arr = request->arr;
        int batch_index = request->index;
	      boost::latch * gen_latch = request->gen_latch;
        //std::cout << "process " << filename << "index" << batch_index;
        //auto time_now = boost::posix_time::microsec_clock::universal_time();
        Mat image;
        image = imread(filename, CV_LOAD_IMAGE_COLOR);   // Read the file
        if(! image.data )                              // Check for invalid input
        {
            std::cout <<  "Could not open or find the image" << std::endl ;
            return;
        }
        Mat resized_image;
        cv::resize(image,resized_image,Size(224,224));
        Mat resized_image_fp32;
        resized_image.convertTo(resized_image_fp32,CV_32FC3);
        std::vector<Mat> bgr_planes;
        split(resized_image_fp32,bgr_planes);
        // auto time_now1 = boost::posix_time::microsec_clock::universal_time();
        // auto time_elapse = time_now1 - time_now;
        // int ticks = time_elapse.ticks();
        // std::cout << filename <<":"<< ticks << std::endl;
        //from caffe code 
        int channel = 3;
        int width = 224;
        int height = 224;
        float * _arr = arr + batch_index * 3 * 224 * 224;
        for (auto c = 0 ; c < channel; c++) {
            float * dst_addr = _arr + c*width * height;
            std::memcpy(dst_addr,bgr_planes[c].data,sizeof (float) *  width * height);
        }
	      gen_latch->count_down();
    }
}

int main(int argc, char ** argv)
{

    Py_Initialize();
		np::initialize();
    int batch_per_graph = 4;
    int replication_factor = 8;
    int gradient_accl_factor = 16;
    int batches_per_step = 16;
    int channel = 3;
    int width = 224;
    int height = 224;
    if( argc != 2)
    {
     cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
     return -1;
    }
    char * imgFilesList = argv[1];

    int num_worker = 48;
    std::vector<boost::thread *> threads;


    for (auto i = 0; i <num_worker; i++ ) {
        threads.push_back(new boost::thread(readimg_thread));
    }
    //put pair into blocking queue

    int total_size = batch_per_graph * replication_factor * gradient_accl_factor * batches_per_step * channel * width * height;
    float * arr = new float[total_size];
    std::string file;
    ifstream myfile(imgFilesList);
    
    int total_batch_size = batch_per_graph * replication_factor * gradient_accl_factor * batches_per_step;
    auto start_time = boost::posix_time::microsec_clock::universal_time();

    boost::latch* gen_latch = new boost::latch(total_batch_size);
    int i = 0;
    while (getline(myfile,file)) {
        imgReadQueue.push(new imgFileReadRequest {file,arr,i,gen_latch});
        i++;
    }
    if (! gen_latch->try_wait())
      if (gen_latch->wait_for(boost::chrono::milliseconds(100)) ==  boost::cv_status::timeout)
  	    if (gen_latch->wait_until(boost::chrono::steady_clock::now()+boost::chrono::milliseconds(100)) ==  boost::cv_status::timeout)
          gen_latch->wait();

    int first_dim = batch_per_graph * replication_factor * gradient_accl_factor * batches_per_step;
    auto shape = p::make_tuple(first_dim,
                  channel,
                  height,
                  width
                );
    auto stride = p::make_tuple(width * height * channel,
                  width * height,
                  width,
                  1) ;
    np::dtype dt1 = np::dtype::get_builtin<float>();
    auto mul_data_ex = np::from_data(arr,
                    dt1,
                    shape,
                    stride,
                    p::object());
    
    
    auto shape_l = p::make_tuple(batches_per_step,
                  gradient_accl_factor,
                  replication_factor,
                  batch_per_graph,
                  channel,
                  height,
                  width
                ); 
    
    mul_data_ex = mul_data_ex.reshape(shape_l);
//   np::dtype dt_float32 = np::dtype::get_builtin<float>();

    auto time_before_cast = boost::posix_time::microsec_clock::universal_time();
//    mul_data_ex = mul_data_ex.astype(dt_float32);
    auto time_after_cast = boost::posix_time::microsec_clock::universal_time();
    auto elapsed_cast = time_after_cast - time_before_cast;

    auto time_now = boost::posix_time::microsec_clock::universal_time();
    auto time_elapse = time_now - start_time;
    int ticks = time_elapse.ticks();
    int ticks_cast = elapsed_cast.ticks();


    std::cout << "Total Cost:"<< ticks << std::endl;
    std::cout << "Total Cost for elapsed_cast:"<< ticks_cast << std::endl;
    std::cout << "Selective multidimensional array :: "<<std::endl
      << p::extract<char const *>(p::str(mul_data_ex)) << std::endl ;
    delete gen_latch;
  
}


