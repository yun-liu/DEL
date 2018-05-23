#ifndef CAFFE_EGB_SEGMENT_LAYER_HPP_
#define CAFFE_EGB_SEGMENT_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

template <typename Dtype>
class EgbSegmentLayer : public Layer<Dtype> {
public:
  explicit EgbSegmentLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){}
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& top){}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& top);

  int height_;
  int width_;
  int channels_;
  float bound;
  int min_size;

};

}

#endif
