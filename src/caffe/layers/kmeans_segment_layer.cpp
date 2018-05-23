#include <algorithm>
#include <vector>
#include <fstream>

#include "caffe/layers/kmeans_segment_layer.hpp"
#include "caffe/segment/kmeans.hpp"

namespace caffe {

template <typename Dtype>
void KmeansSegmentLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
    const KmeansSegmentParameter& kmeans_segment_param = this->layer_param_.kmeans_segment_param();
    CHECK(kmeans_segment_param.has_num_part())
        << "Number of regions to be segmented must be specified.";
    K = kmeans_segment_param.num_part();
    CHECK_EQ(bottom[0]->num(), 1) << "Only support single image";
}

template <typename Dtype>
void KmeansSegmentLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
    vector<int> new_shape(bottom[0]->shape());
    new_shape[1] = 3;
    top[0]->Reshape(new_shape);

    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
}

template <typename Dtype>
void KmeansSegmentLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

    int channels = bottom[0]->shape(1);
    image<feature>* im = new image<feature>(width_, height_);
    image<rgb>* im_seg = new image<rgb>(width_, height_);
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();

    for(int i = 0; i < height_; i++) {
      for(int j = 0; j < width_; j++) {
        for(int c = 0; c < channels; c++) {
          imRef(im, j, i).data[c] = bottom_data[(c * height_ + i) * width_ + j];
        }
      }
    }
    Kmeanseg segment;
    im_seg = segment.PerformKmeans_ForGivenK(im, K);

    for(int i = 0; i < height_; i++) {
      for(int j = 0; j < width_; j++) {
        top_data[(0 * height_ + i) * width_ + j] = imRef(im_seg, j, i).r;
        top_data[(1 * height_ + i) * width_ + j] = imRef(im_seg, j, i).g;
        top_data[(2 * height_ + i) * width_ + j] = imRef(im_seg, j, i).b;
      }
    }
}

template <typename Dtype>
void KmeansSegmentLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

INSTANTIATE_CLASS(KmeansSegmentLayer);
REGISTER_LAYER_CLASS(KmeansSegment);
} //namespace caffe
