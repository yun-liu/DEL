#ifndef CAFFE_KMEAN_SEGMENT_LAYER_HPP_
#define CAFFE_KMEAN_SEGMENT_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>

class KmeansSegmentLayer : public Layer<Dtype> {
public:
  explicit KmeansSegmentLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& top);

  int height_;
  int width_;
  int K;   // given K

};

}

#endif
