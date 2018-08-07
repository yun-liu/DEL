#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cassert>

#include "caffe/layers/image_labelmap_data_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
ImageLabelmapDataLayer<Dtype>::~ImageLabelmapDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void ImageLabelmapDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();

  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string img_filename;
  string gt_filename;
  string superpixelabel_filename;
  while (infile >> img_filename >> gt_filename >> superpixelabel_filename) {
    lines_.push_back(std::make_pair(std::make_pair(img_filename, gt_filename), superpixelabel_filename));
  }

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first.first,
                                    new_height, new_width, is_color);

  cv::Mat cv_gt = ReadImageToCVMat(root_folder + lines_[lines_id_].first.second,
                                    new_height, new_width, 0);

  cv::Mat cv_splabel = cv::imread(root_folder + lines_[lines_id_].second, CV_LOAD_IMAGE_ANYDEPTH);



  //const int channels = cv_img.channels();
  const int height = cv_img.rows;
  const int width = cv_img.cols;

  const int gt_channels = cv_gt.channels();
  const int gt_height = cv_gt.rows;
  const int gt_width = cv_gt.cols;

  const int splabel_channels = cv_splabel.channels();
  const int splabel_height = cv_splabel.rows;
  const int splabel_width = cv_splabel.cols;
  CHECK((height == gt_height) && (width == gt_width)) << "groundtruth size != image size";
  CHECK((height == splabel_height) && (width == splabel_width)) << "superpixel image size != image size";
  CHECK((gt_channels == 1) && (splabel_channels == 1)) << "GT image and superpixel image channel number should be 1";
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first.first;

  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  vector<int> top_shape_labelmap = this->data_transformer_->InferBlobShape(cv_gt);
//  vector<int> top_shape_splabelmap(top_shape_labelmap);

  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_EQ(batch_size, 1) << "Batch size must equal to 1...";

  top_shape[0] = batch_size;
  top_shape_labelmap[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
    this->prefetch_[i].labelmap_.Reshape(top_shape_labelmap);
    this->prefetch_[i].superpixelabelmap_.Reshape(top_shape_labelmap);
  }

  top[0]->Reshape(top_shape);
  top[1]->Reshape(top_shape_labelmap);
  top[2]->Reshape(top_shape_labelmap);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  LOG(INFO) << "output label size: " << top[1]->num() << ","
      << top[1]->channels() << "," << top[1]->height() << ","
      << top[1]->width();
  LOG(INFO) << "output superpixel label size: " << top[2]->num() << ","
      << top[2]->channels() << "," << top[2]->height() << ","
      << top[2]->width();
}

template <typename Dtype>
void ImageLabelmapDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void ImageLabelmapDataLayer<Dtype>::load_batch(LabelmapBatch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(batch->labelmap_.count());
  CHECK(batch->superpixelabelmap_.count());

  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const int crop_size = image_data_param.crop_size();
  const bool is_color = image_data_param.is_color();
  const string root_folder = image_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first.first,
      new_height, new_width, is_color);

  //cv::Mat cv_gt = ReadImageToCVMat(root_folder + lines_[lines_id_].first.second,
  //    new_height, new_width, 0);
  cv::Mat cv_gt = cv::imread(root_folder + lines_[lines_id_].first.second, CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_splabel = cv::imread(root_folder + lines_[lines_id_].second, CV_LOAD_IMAGE_ANYDEPTH);

  //std::cout << lines_[lines_id_].second << std::endl;
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first.first;

  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  vector<int> top_shape_labelmap = this->data_transformer_->InferBlobShape(cv_gt);

  batch->data_.Reshape(top_shape);
  batch->labelmap_.Reshape(top_shape_labelmap);
  batch->superpixelabelmap_.Reshape(top_shape_labelmap);

  // datum scales
  const int lines_size = lines_.size();
  timer.Start();
  CHECK_GT(lines_size, lines_id_);

  const int height = cv_img.rows;
  const int width = cv_img.cols;

  const int gt_channels = cv_gt.channels();
  const int gt_height = cv_gt.rows;
  const int gt_width = cv_gt.cols;

  const int splabel_channels = cv_splabel.channels();
  const int splabel_height = cv_splabel.rows;
  const int splabel_width = cv_splabel.cols;

  CHECK((height == gt_height) && (width == gt_width)) << "groundtruth size != image size";
  CHECK((height == splabel_height) && (width == splabel_width)) << "superpixel image size != image size";
  CHECK((gt_channels == 1) && (splabel_channels == 1)) << "GT image and superpixel image channel number should be 1";

  read_time += timer.MicroSeconds();
  timer.Start();

  int h_off = 0;
  int w_off = 0;
  float im_scale = -1;
  bool do_mirror = false;
  this->data_transformer_->LocTransform(cv_img, &(batch->data_), h_off, w_off, do_mirror, im_scale, crop_size);
  this->data_transformer_->LabelmapTransform(cv_gt, &(batch->labelmap_), h_off, w_off, do_mirror, im_scale, crop_size);
  this->data_transformer_->SuperpixelLabelmapTransform(cv_splabel, &(batch->superpixelabelmap_), h_off, w_off, do_mirror, im_scale, crop_size);

  //const Dtype* sp = batch->labelmap_.cpu_data();
  //const Dtype* max_label = std::max_element(sp, sp+height*width);
  //int numsp_ = static_cast<int>(*max_label) + 1;  //number of superpixel
  //std::cout << "first: " << height << " " <<  width << " " << numsp_ << " " << batch->labelmap_.count() << std::endl;

  trans_time += timer.MicroSeconds();

  lines_id_++;     // go to the next iter
  if (lines_id_ >= lines_size) {
    // We have reached the end. Restart from the first.
    DLOG(INFO) << "Restarting data prefetching from start.";
    lines_id_ = 0;
    if (this->layer_param_.image_data_param().shuffle()) {
      ShuffleImages();
    }
  }

  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageLabelmapDataLayer);
REGISTER_LAYER_CLASS(ImageLabelmapData);

}  // namespace caffe
