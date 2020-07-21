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

//typedef std::pair<std::vector<int>::iterator,std::vector<int>::iterator> ve_iterator;
class ImagenetDataset{
	public:
		ImagenetDataset(std::string imageFolder,
						int epochs,
						int batch_per_graph,
						int replication_factor,
						int gradient_accl_factor,
						int batches_per_step,
						int num_workers,
						float prefetch_ratio,
						bool is_training,
						bool drop = true,
						bool shuffle = true,
						bool synthetic = false,
						int channel = 3,
						int height = 224,
						int width = 224){
			
			Py_Initialize();
			np::initialize();

			this->batch_per_graph = batch_per_graph;
			this->replication_factor = replication_factor;
			this->gradient_accl_factor = gradient_accl_factor;
			this->batches_per_step = batches_per_step;

			this->epochs = epochs;
			this->imageFolder = imageFolder;
			this->num_workers = num_workers;
			this->prefetch_ratio;
			this->total_batch_size = batch_per_graph * replication_factor * gradient_accl_factor * batches_per_step;
			this->width = width;
			this->height = height;
			this->channel = channel;


			this->train_folder = imageFolder + "/train";
			this->val_folder	 = imageFolder + "/val";
			if (!fs::exists(imageFolder)) {
				std::cout << "The ImageNet folder:  " << imageFolder << " is not valid" << std::endl;

			}	
			if (!fs::is_directory(imageFolder)){
				std::cout << "The ImageNet folder:  " << imageFolder << " is not a directory" << std::endl;
			}
			if (!fs::exists(train_folder)) {
				std::cout << "The ImageNet subfolder : " << train_folder << " does not exists" << std::endl;
			}

			this->total_images =  get_total_train_image();
			this->steps_per_epochs = this->total_images / this->total_batch_size;
			std::cout << "Dataset Information:	" << std::endl;
			std::cout << "\tImageFolder:            " << imageFolder << std::endl;
			std::cout << "\tbatch_per_graph:        " << batch_per_graph << std::endl;
			std::cout << "\treplication_factor:     " << replication_factor << std::endl;
			std::cout << "\tgradient_accl_factor:   " << gradient_accl_factor << std::endl;
			std::cout << "\tbatches_per_step:       " << batches_per_step << std::endl;
			std::cout << "\tnum_workers:            " << num_workers << std::endl;
			std::cout << "\tprefetch_ratio:         " << prefetch_ratio << std::endl;
			std::cout << "\ttotal_images:           " << this->total_images << std::endl;
			std::cout << "\ttotal_batch_size:       " << this->total_batch_size << std::endl;
			std::cout << "\tsteps_per_epochs:       " << this->steps_per_epochs << std::endl;

			fill_batch_indices();
			fill_dataset();	
		}

		void set_num_workers(int num) {
			this->num_workers = num;
		}
		int get_num_workers() {
			return this->num_workers;
		}

		int __len__() {
			return this->steps_per_epochs;
		}

		int get_total_train_image() {
			unsigned long file_count = 0;
			fs::directory_iterator end_iter;
			for (fs::directory_iterator dir_itr(this->train_folder);
          		dir_itr != end_iter;
          		++dir_itr ) {
				if (!fs::is_directory(dir_itr->status())) {
					continue;
				}
				fs::directory_iterator sub_end_iter;
				for (fs::directory_iterator subdir_itr(dir_itr->path().string());
					subdir_itr != sub_end_iter;
          			++subdir_itr) {
					if (fs::is_regular_file( subdir_itr->status())) {
						std::string filename = subdir_itr->path().string();
					
						auto extension = getFileExtension(filename);
						if (extension == ".jpeg" 
							|| extension == ".JPEG") {
							this->filenames.push_back(filename);
							file_count++;
						}
					}
				}
			}
			return file_count;
		}

		void pick_batch_indices(std::set<int> &origin_indice, std::vector<int> &output_indice) {
			std::vector<int> _tmp;
			_tmp.assign(origin_indice.begin(),origin_indice.end());

			boost::random::mt19937 rng; 
			std::set<int> rand_indices;
			boost::random::uniform_int_distribution<int> indice(0,origin_indice.size() - 1);
			while (rand_indices.size() < this->total_batch_size) {
				auto index = indice(rng);
				rand_indices.insert(index);
			}
			for (auto itr = rand_indices.begin(); itr != rand_indices.end() ;itr++) {
				
				output_indice.push_back(_tmp[*itr]);
				origin_indice.erase(_tmp[*itr]);
			}
		}

		void fill_batch_indices() {
			std::set<int> images_indices;
			for (auto i = 0; i < this->total_images; i++) {
				images_indices.insert(i);
			}
			for (auto i = 0; i < 1; i++) {
//			for (auto i = 0; i < this->epochs; i++) {	
				std::vector<std::vector<int>> steps_indices;
				for (auto j = 0; j < this->steps_per_epochs; j++) {
					std::set<int> filepath_indexes(images_indices);
					
					std::vector<int> batch_indices;
					pick_batch_indices(filepath_indexes,batch_indices);
					steps_indices.push_back(batch_indices);
				}
				this->epoch_batch_indices.push_back(steps_indices);
			}
		}
		
		void readimg(std::string filename, uint8_t *arr, int batch_index) {
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


			uint8_t * _arr = arr + batch_index * this->channel * this->width * this->height;
			for (auto c = 0 ; c < channel; c++) {
				uint8_t * dst_addr = _arr + c*width * height;
				std::memcpy(dst_addr,bgr_planes[c].data,sizeof (uint8_t) *  width * height);
			}

		}

		//Test only
		void read_batch_files(int epoch_id, int step_id) {


			auto time_now = boost::posix_time::microsec_clock::universal_time();

			int total_size = this->total_batch_size * this->channel * this->height * this->width;
			uint8_t *arr = new uint8_t[total_size];
			for (auto i = 0;i < this->total_batch_size; i++) {
				readimg (this->filenames[epoch_batch_indices[epoch_id][step_id][i]], arr, i);
			}

			int first_dim = this->batch_per_graph * this->replication_factor * this->gradient_accl_factor * this->batches_per_step;
			auto shape = p::make_tuple(first_dim,
										this->channel,
										this->height,
										this->width
									);
			auto stride = p::make_tuple(this->width * this->height * this->channel,
										this->width * this->height,
										this->width,
										1) ;
			np::dtype dt1 = np::dtype::get_builtin<uint8_t>();
			auto mul_data_ex = np::from_data(arr,
											dt1,
											shape,
											stride,
											p::object());

			auto shape_l = p::make_tuple(this->batches_per_step,
										this->gradient_accl_factor,
										this->replication_factor,
										this->batch_per_graph,
										this->channel,
										this->height,
										this->width
									); 
			mul_data_ex = mul_data_ex.reshape(shape_l);

			np::dtype dt_float32 = np::dtype::get_builtin<float>();
    		mul_data_ex = mul_data_ex.astype(dt_float32);
			auto time_now1 = boost::posix_time::microsec_clock::universal_time();
			auto time_elapse = time_now1 - time_now;
			int ticks = time_elapse.ticks();
			std::cout << "Concat time cost:" << ticks << std::endl;

			std::cout << "Selective multidimensional array :: "<<std::endl
            	<< p::extract<char const *>(p::str(mul_data_ex)) << std::endl ;
		}

		std::string getFileExtension(std::string filePath)
		{
			// Create a Path object from given string
			fs::path pathObj(filePath);
			// Check if file name in the path object has extension
			if (pathObj.has_extension()) {
				// Fetch the extension from path object and return
				return pathObj.extension().string();
			}
			// In case of no extension return empty string
			return "";
		}

		void fill_dataset() {
			for (auto i = 0; i < this->steps_per_epochs; i++ ) {

				np::dtype dtype_float32 = np::dtype::get_builtin<float>();
				np::dtype dtype_int32 = np::dtype::get_builtin<int>();
				p::tuple image_shape = p::make_tuple(this->batches_per_step,
													  this->gradient_accl_factor,
													  this->replication_factor,
													  this->batch_per_graph,
													  3,
													  224,
													  224);
				p::tuple batch_shape = p::make_tuple(this->batches_per_step,
													  this->gradient_accl_factor,
													  this->replication_factor,
													  this->batch_per_graph);									  
				np::ndarray image = np::zeros(image_shape, dtype_float32);
				np::ndarray label = np::zeros(batch_shape, dtype_int32);

				p::tuple elem = p::make_tuple(image,label);
				this->steps_data.push_back(elem);
			}

		}

		std::vector<p::tuple>::iterator begin() { return this->steps_data.begin();}
    	std::vector<p::tuple>::iterator end()   { return this->steps_data.end();}

		void print_batch_indice(int epoch_id, int step_id) {
			for (auto i = 0;i < this->total_batch_size; i++) {
				std::cout << this->filenames[epoch_batch_indices[epoch_id][step_id][i]] << std::endl;
			}	
		}

	private:
		std::string imageFolder;
		std::string train_folder;
		std::string val_folder;
		int prefetch_ratio;
		int total_batch_size;
		int num_workers;
		int steps_per_epochs;
		int total_images;
		int epochs;
		int batches_per_step;
		int batch_per_graph;
		int gradient_accl_factor;
		int replication_factor;
		int width;
		int height;
		int channel;
		std::vector<std::string> filenames;
		std::vector<std::vector<std::vector<int>>> epoch_batch_indices;
		std::vector<p::tuple> steps_data;
};


BOOST_PYTHON_MODULE(dataset)
{
	    using namespace boost::python;
	    class_<ImagenetDataset>("ImagenetDataset",init<std::string, int, int, int, int, int, int, float, bool>())
			.def("__iter__", range(&ImagenetDataset::begin, &ImagenetDataset::end))
			.def("__len__",  &ImagenetDataset::__len__)
			.def("print_batch_indice",  &ImagenetDataset::print_batch_indice)
			.def("read_batch_files",  &ImagenetDataset::read_batch_files)
		    .add_property("num_fetch_workers", &ImagenetDataset::get_num_workers, &ImagenetDataset::set_num_workers);
}
