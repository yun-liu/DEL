#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>
#include <cstdlib>

#include "caffe/layers/similarity_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
// bottom[0]: superpixel pooled feature map    1 * 1 * S * 64
// bottom[1]: superpixel segmented labels      1 * 1 * 1 * S
// bottom[2]: superpixel label
template <typename Dtype>
inline Dtype sigmoid2(Dtype x) {
  return 0.5 * tanh(-0.5 * x) + 0.5;
}

template <typename Dtype>
void SimilarityLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  const SimilarityLossParameter& similarity_loss_param = this->layer_param_.similarity_loss_param();
  CHECK((similarity_loss_param.has_sample_points()))
      << "Either axis or concat_dim should be specified; not both.";
  num_per_region_ = similarity_loss_param.sample_points();

  has_ignore_label_ = this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
    std::cout << "which to ignore:" << ignore_label_ << std::endl;
  }
  rng_.reset(new Caffe::RNG(caffe_rng_rand()));
}

template <typename Dtype>
void SimilarityLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  outer_num_ = bottom[0]->count(0, 1);
  inner_num_ = bottom[0]->shape(2);
  channels_ = bottom[0]->shape(3);

  height_ = bottom[2]->shape(2);
  width_ = bottom[2]->shape(3);

  CHECK_EQ(inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if prediction shape is (1, 1, S, W), "
      << "label count (number of labels) must be 1*1*1*S,";
}

template <typename Dtype>
void SimilarityLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    all_labels.clear();

    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_label = bottom[1]->cpu_data();
    const Dtype* bottom_sp_label = bottom[2]->cpu_data();

    const int dx4[4] = {-1, 0, 1, 0};
  	const int dy4[4] = { 0,-1, 0, 1};

    for(int i = 0; i < outer_num_; i++) {
      vector<vector<int> > adjacent(inner_num_);
      for(int j = 0; j < height_; j++) {
        for(int k = 0; k < width_; k++) {
          int curr = bottom_sp_label[i*height_*width_ + j*width_ + k];
          for(int c = 0; c < 4; c++) {
            int nx = k + dx4[c], ny = j + dy4[c];   // move to 8 directions
            if(nx >= 0 && nx < width_ && ny >= 0 && ny < height_) {
              int pre = bottom_sp_label[i*height_*width_ + ny*width_ + nx];
              if(curr > pre) {
                int la1 = static_cast<int>(bottom_label[curr]);
                int la2 = static_cast<int>(bottom_label[pre]);
                if(la1 == -1 || la2 == -1) continue;
                vector<int>::iterator iter = find(adjacent[curr].begin(), adjacent[curr].end(), pre);
                if (iter == adjacent[curr].end()) {
                  adjacent[curr].push_back(pre);
                }
              }
            }
          }
        }
      }
      all_labels.push_back(adjacent);
    }

    counter_pos_ = counter_neg_ = 0;
    Dtype total_loss = 0;

    Blob<Dtype> scale;
    Blob<Dtype> sub_scale;

    for(int i = 0; i < outer_num_; i++) {

      const int S = inner_num_;
      Dtype loss_pos = 0, loss_neg = 0;
      vector<vector<int> >& adjacent = all_labels[i];

      for(int n = 0; n < S; n++) {
        vector<int> currsp = adjacent[n];
        int adjsz = currsp.size();
        int lf = static_cast<int>(bottom_label[i*S + n]);   // if label == -1 go to next superpixel

        if(adjsz > 0) {
          scale.Reshape(adjsz+1, channels_, 1, 1);
          sub_scale.Reshape(adjsz, channels_, 1, 1);
          Dtype* scale_data = scale.mutable_cpu_data();
          Dtype* sub_scale_data = sub_scale.mutable_cpu_data();
          caffe_set((adjsz+1)*channels_, Dtype(0), scale_data);

          for(int k = 0; k < channels_; k++) {
            scale_data[k] = bottom_data[i*S*channels_ + n*channels_ + k];
          }
          for(int j = 0; j < adjsz; j++) {
            for(int k = 0; k < channels_; k++) {
              scale_data[(j+1)*channels_ + k] = bottom_data[i*S*channels_ + currsp[j]*channels_ + k];
            }
          }
          // ep - eq
          for(int k = 0; k < adjsz; k++) {
            caffe_sub(channels_, scale_data, scale_data + (k+1)*channels_, sub_scale_data + k*channels_);
          }
          caffe_abs(adjsz*channels_, sub_scale_data, sub_scale_data);
          for(int k = 0; k < adjsz; k++) {
            int ls = static_cast<int>(bottom_label[i*S + currsp[k]]);
            Dtype sim = caffe_cpu_asum(channels_, sub_scale_data + k*channels_);
            Dtype similarity = Dtype(2) * sigmoid2(sim), logsim;

            if(lf == ls) {
              if(!isinf(logsim = log(similarity))) {
                loss_pos -= logsim;
                counter_pos_++;
              }
            }
            else if(!isinf(logsim = log(1 - similarity))) {
                loss_neg -= logsim;
                counter_neg_++;
            }
          }
        }
      }
      total_loss += (loss_pos*counter_neg_ + loss_neg*counter_pos_) / (counter_pos_+ counter_neg_);
    }
    top[0]->mutable_cpu_data()[0] = total_loss; // / (counter_pos_+counter_neg_);
}

template <typename Dtype>
void SimilarityLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_label = bottom[1]->cpu_data();

    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_set(bottom[0]->count(), Dtype(0.), bottom_diff);

    Blob<Dtype> scale;
    Blob<Dtype> sub_scale;
    Blob<Dtype> abs_sub;
    Blob<Dtype> dim_diff;

    for(int i = 0; i < outer_num_; i++) {

      const int S = inner_num_;
      vector<vector<int> >& adjacent = all_labels[i];
      for(int n = 0; n < S; n++) {   // for every superpixel and his neighborhood
        int lf = static_cast<int>(bottom_label[i*S + n]);   // if label == -1 go to next superpixel
        vector<int>& currsp = adjacent[n];
        int adjsz = currsp.size();
        if(adjsz > 0) {

          scale.Reshape(adjsz+1, channels_, 1, 1);
          sub_scale.Reshape(adjsz, channels_, 1, 1);
          abs_sub.Reshape(adjsz, channels_, 1, 1);
          dim_diff.Reshape(channels_, 1, 1, 1);

          Dtype* scale_data = scale.mutable_cpu_data();
          Dtype* sub_scale_data = sub_scale.mutable_cpu_data();
          Dtype* abs_sub_data = abs_sub.mutable_cpu_data();
          Dtype* dim_diff_data = dim_diff.mutable_cpu_data();
          caffe_set((adjsz+1)*channels_, Dtype(0), scale_data);

          for(int k = 0; k < channels_; k++) {
            scale_data[k] = bottom_data[i*S*channels_ + n*channels_ + k];
          }
          for(int j = 0; j < adjsz; j++) {
            for(int k = 0; k < channels_; k++) {
              scale_data[(j+1)*channels_ + k] = bottom_data[i*S*channels_ + currsp[j]*channels_ + k];
            }
          }
          caffe_set(channels_, Dtype(0.), dim_diff_data);

          for(int k = 0; k < adjsz; k++) {
            caffe_sub(channels_, scale_data, scale_data + (k+1)*channels_, sub_scale_data + k*channels_);
          }
          caffe_abs(adjsz*channels_, sub_scale_data, abs_sub_data);

          for(int k = 0; k < adjsz; k++) {
            int ls = static_cast<int>(bottom_label[i*S + currsp[k]]);

            Dtype sim = 0, base_diff = 0;
            sim = caffe_cpu_asum(channels_, abs_sub_data + k*channels_);

            Dtype similarity = Dtype(2) * sigmoid2(sim);
            if((lf == ls && isinf(log(similarity))) || (lf != ls && isinf(log(1 - similarity)))
            || (lf == ls && similarity == 1) || (lf != ls && (1 - similarity) == 1))
              continue;

            if(lf == ls) {
              base_diff = Dtype(1) / (Dtype(1) + exp(-sim)) * counter_neg_ / (counter_neg_ + counter_pos_);
            }
            else {
              base_diff = -Dtype(2) / (exp(sim) - exp(-sim)) * counter_pos_ / (counter_neg_ + counter_pos_);
            }
            for(int c = 0; c < channels_; c++) {
              if (sub_scale_data[k*channels_ + c] < 0) {
                dim_diff_data[c] = base_diff * Dtype(-1);
              }
              else {
                dim_diff_data[c] = base_diff;
              }
            }

            for(int c = 0; c < channels_; c++) {
              bottom_diff[i*S*channels_ + n*channels_ + c] += dim_diff_data[c];
              bottom_diff[i*S*channels_ + currsp[k]*channels_ + c] -= dim_diff_data[c];
            }
          }
        }
      }
    }
    // Scale down gradient
    Dtype loss_weight = top[0]->cpu_diff()[0]; // / (counter_pos_+counter_neg_);
    caffe_scal(bottom[0]->count(), loss_weight, bottom_diff);
  }

}

#ifdef CPU_ONLY
STUB_GPU(SimilarityLossLayer);
#endif

INSTANTIATE_CLASS(SimilarityLossLayer);
REGISTER_LAYER_CLASS(SimilarityLoss);

}  // namespace caffe
