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
    uint8_t *arr; 
    int index;
};

BlockingQueue<imgFileReadRequest*> imgReadQueue;

void readimg_thread() {

    std::cout << "starting thread " << std::endl;
    while (1) {
        auto request = imgReadQueue.pop();
        auto filename = request->imgFilePath;
        uint8_t * arr = request->arr;
        int batch_index = request->index;

        std::cout << "process " << filename << "index" << batch_index;

        auto time_now = boost::posix_time::microsec_clock::universal_time();
        Mat image;
        image = imread(filename, CV_LOAD_IMAGE_COLOR);   // Read the file
        if(! image.data )                              // Check for invalid input
        {
            std::cout <<  "Could not open or find the image" << std::endl ;
            return;
        }
        Mat resized_image;
        cv::resize(image,resized_image,Size(224,224));

        std::vector<Mat> bgr_planes;
        split(resized_image,bgr_planes);


        auto time_now1 = boost::posix_time::microsec_clock::universal_time();
        auto time_elapse = time_now1 - time_now;
        int ticks = time_elapse.ticks();
        std::cout << filename <<":"<< ticks << std::endl;
        //from caffe code 

        int channel = 3;
        int width = 224;
        int height = 224;
        uint8_t * _arr = arr + batch_index * 3 * 224 * 224;
        for (auto c = 0 ; c < channel; c++) {
            uint8_t * dst_addr = _arr + c*width * height;
            std::memcpy(dst_addr,bgr_planes[c].data,sizeof (uint8_t) *  width * height);
        }
    }
}

int main(int argc, char ** argv)
{

    int batch_per_graph = 4;
    int replication_factor = 8;
    int gradient_accl_factor = 8;
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

    int num_worker = 96;
    std::vector<boost::thread *> threads;


    for (auto i = 0; i <num_worker; i++ ) {
        threads.push_back(new boost::thread(readimg_thread));
    }
    //put pair into blocking queue

    int total_size = batch_per_graph * replication_factor * gradient_accl_factor * batches_per_step * channel * width * height;
    uint8_t * arr = new uint8_t[total_size];

    std::string file;
    ifstream myfile(imgFilesList);
    int i = 0;

    while (getline(myfile,file)) {
        std::cout << file << std::endl; 
        for (auto j = 0; j < 300; j++) 
        imgReadQueue.push(new imgFileReadRequest {file,arr,i});
        i++;
    }
  
    boost::this_thread::sleep_for(boost::chrono::milliseconds(1000000));
}


