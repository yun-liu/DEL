#ifndef CAFFE_SIMILARITY_LOSS_LAYER_HPP_
#define CAFFE_SIMILARITY_LOSS_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
class SimilarityLossLayer : public LossLayer<Dtype> {
 public:
  explicit SimilarityLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "SimilarityLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int numsp_;
  int numfeat_;
  int height_;
  int width_;
  int num_per_region_;
  int counter_pos_, counter_neg_;
  vector<vector<vector<int> > > all_labels;
  shared_ptr<Caffe::RNG> rng_;
};

}  // namespace caffe
#endif  // CAFFE_SIMILARITY_LOSS_LAYER_HPP_
