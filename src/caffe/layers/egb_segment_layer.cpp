#include <algorithm>
#include <vector>
#include <fstream>
#include <ctime>
#include "caffe/layers/egb_segment_layer.hpp"

namespace caffe {

template <typename Dtype>
void EgbSegmentLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
    const EgbSegmentParameter& egb_segment_param = this->layer_param_.egb_segment_param();
    CHECK(egb_segment_param.has_bound() && egb_segment_param.has_min_size())
        << "Minimum size of region and constant restrain must be specified.";

    bound = egb_segment_param.bound();
    min_size = egb_segment_param.min_size();

    CHECK_EQ(bottom[0]->num(), 1) << "Only support single image";
}

template <typename Dtype>
void EgbSegmentLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
    vector<int> new_shape(bottom[1]->shape());
    top[0]->Reshape(new_shape);
    height_ = bottom[1]->height();
    width_ = bottom[1]->width();
}

INSTANTIATE_CLASS(EgbSegmentLayer);
REGISTER_LAYER_CLASS(EgbSegment);

} //namespace caffe
