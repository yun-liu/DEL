#include <cmath>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/similarity_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

  template <typename TypeParam>
  class SimilarityLossLayerTest : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;

   protected:
    SimilarityLossLayerTest()
        : blob_bottom_data_(new Blob<Dtype>(1, 5, 20, 20)),
          blob_bottom_label_(new Blob<Dtype>(1, 1, 20, 20)),
          blob_top_loss_(new Blob<Dtype>()) {
      // fill the values
      FillerParameter filler_param;
      filler_param.set_std(10);
      GaussianFiller<Dtype> filler(filler_param);
      filler.Fill(this->blob_bottom_data_);
      blob_bottom_vec_.push_back(blob_bottom_data_);

      int onum = blob_bottom_data_->shape(0);
      int channels = blob_bottom_data_->shape(1);
      int pnum = blob_bottom_label_->count(2);
      for (int i = 0; i < blob_bottom_label_->count(); ++i) {
        blob_bottom_label_->mutable_cpu_data()[i] = caffe_rng_rand() % 5;
      }
      for (int i = 0; i < onum; ++i) {
        for(int j = 0; j < pnum; j++) {
          int label = blob_bottom_label_->cpu_data()[i * pnum + j];
          for(int c = 0; c < channels; c++) {
            blob_bottom_data_->mutable_cpu_data()[i * pnum * channels + c * pnum + j] = 0.01 * label + (caffe_rng_rand() % 2) / Dtype(5);
          }
        }
      }
      blob_bottom_vec_.push_back(blob_bottom_label_);
      blob_top_vec_.push_back(blob_top_loss_);
     }
    virtual ~SimilarityLossLayerTest() {
      delete blob_bottom_data_;
      delete blob_bottom_label_;
      delete blob_top_loss_;
    }

    Blob<Dtype>* const blob_bottom_data_;
    Blob<Dtype>* const blob_bottom_label_;
    Blob<Dtype>* const blob_top_loss_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SimilarityLossLayerTest, TestDtypesAndDevices);
/*
TYPED_TEST(SimilarityLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SimilarityLossParameter* similarity_loss_param = layer_param.mutable_similarity_loss_param();
  similarity_loss_param->set_sample_points(2);
  SimilarityLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}
*/

TYPED_TEST(SimilarityLossLayerTest, TestForwardIgnoreLabel) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  SimilarityLossParameter* similarity_loss_param = layer_param.mutable_similarity_loss_param();
  similarity_loss_param->set_sample_points(2);
  // First, compute the loss with all labels
  scoped_ptr<SimilarityLossLayer<Dtype> > layer(
      new SimilarityLossLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype full_loss = this->blob_top_loss_->cpu_data()[0];
  // Now, accumulate the loss, ignoring each label in {0, ..., 4} in turn.
  Dtype accum_loss = 0;
  for (int label = 0; label < 5; ++label) {
    layer_param.mutable_loss_param()->set_ignore_label(label);
    layer.reset(new SimilarityLossLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    accum_loss += this->blob_top_loss_->cpu_data()[0];
    std::cout << "accum_loss: " << accum_loss << std::endl;
  }
  // Check that each label was included all but once.
  EXPECT_NEAR(4 * full_loss, accum_loss, 1e-4);
}

}
