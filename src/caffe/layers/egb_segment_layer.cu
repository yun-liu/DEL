#include <algorithm>
#include <vector>
#include <fstream>
#include <ctime>
#include <math.h>
#include "caffe/segment/segment-graph.h"
#include "caffe/layers/egb_segment_layer.hpp"
#include "caffe/util/gpu_util.cuh"

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

}

#include <cuda_runtime.h>

template <typename Dtype>
__global__ void SuperpixelPoolForward(const int nthreads,
    const Dtype* const bottom_data, const Dtype* const sp_data, const int height_,
    const int width_, const int channels_, Dtype* sp_feature) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int h = index / width_;
    int w = index % width_;
    int label = static_cast<int>(sp_data[h*width_ + w]);
    for (int i = 0; i < channels_; i++) {
      caffe_gpu_atomic_add(bottom_data[(i*height_ + h)*width_ + w],
          sp_feature + label*(channels_ + 1) + i);
    }
    caffe_gpu_atomic_add(Dtype(1), sp_feature + label*(channels_ + 1) + channels_);
  }
}

template <typename Dtype>
__global__ void AVGFeatureForward(const int nthreads,
    Dtype* const sp_feature, const int channels_) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    for (int i = 0; i < channels_; i++) {
      sp_feature[index*(channels_ + 1) + i] /= sp_feature[index*(channels_ + 1) + channels_];
    }
  }
}

template <typename Dtype>
__global__ void FindNeighborsForward(const int nthreads, const Dtype* const sp_data,
    Dtype* const neighbor, const int numsp_, const int height_, const int width_) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int dx[4] = {-1, 0, 1, 0};
    const int dy[4] = {0, -1, 0, 1};
    const int di[4] = {-1, -width_, 1, width_};
    const int h = index / width_;
    const int w = index % width_;
    int curr = sp_data[index];
    for (int i = 0; i < 4; i++) {
      if (w + dx[i] >= 0 && w + dx[i] < width_ && h + dy[i] >= 0 && h + dy[i] < height_) {
        int pre = static_cast<int>(sp_data[index + di[i]]);
        if (curr > pre) {
          neighbor[curr*numsp_ + pre] = 1;
        }
      }
    }
  }
}

template <typename Dtype>
__global__ void CountNumNeiborForward(const int nthreads, Dtype* const count,
    const Dtype* const neighbor) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    if (neighbor[index] > 0) {
      caffe_gpu_atomic_add(Dtype(1), count);
    }
  }
}

template <typename Dtype>
__global__ void GetNeiborListForward(const int nthreads, Dtype* const count,
    const Dtype* const neighbor, Dtype* const weight, const int numsp_) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    if (neighbor[index] > 0) {
      int idx = caffe_gpu_atomic_add(Dtype(1), count) + 1;
      weight[int(idx*3 + 0)] = index / numsp_;
      weight[int(idx*3 + 1)] = index % numsp_;
    }
  }
}

template <typename Dtype>
__global__ void DissimilarityForward(const int nthreads,
    Dtype* const sp_feature, Dtype* const weight, const int channels_) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype product = 0.f;
    int sp1 = int(weight[index*3 + 0]);
    int sp2 = int(weight[index*3 + 1]);
    for (int i = 0; i < channels_; i++) {
      product += std::abs(sp_feature[sp1*(channels_ + 1) + i]
          - sp_feature[sp2*(channels_ + 1) + i]);
    }
    weight[index*3 + 2] = 1 - 2.0 / (1 + exp(product));
  }
}

template <typename Dtype>
void EgbSegmentLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
    //clock_t start, finish;
    //start = clock();

    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    channels_ = bottom[0]->channels();

    Dtype* bottom_data = bottom[0]->mutable_gpu_data();
    const Dtype* gpu_sp_data = bottom[1]->gpu_data();
    const Dtype* sp_data = bottom[1]->cpu_data();
    bound = bottom[2]->cpu_data()[0];
    min_size = bottom[3]->cpu_data()[0];
    Dtype* top_data = top[0]->mutable_cpu_data();
    const Dtype* max_label = std::max_element(sp_data, sp_data+bottom[1]->count());
    const int numsp_ = static_cast<int>(*max_label) + 1;  //number of superpixel

    vector<int> feature_shape(4, 1);
    feature_shape[2] = numsp_;
    feature_shape[3] = channels_ + 1;
    Blob<Dtype> sp_feature;
    sp_feature.Reshape(feature_shape);
    Dtype* feature = sp_feature.mutable_gpu_data();
    caffe_gpu_set(sp_feature.count(), Dtype(0), feature);

    SuperpixelPoolForward<Dtype><<<CAFFE_GET_BLOCKS(height_*width_),
        CAFFE_CUDA_NUM_THREADS>>>(height_*width_, bottom_data, gpu_sp_data,
        height_, width_, channels_, feature);

    AVGFeatureForward<Dtype><<<CAFFE_GET_BLOCKS(numsp_), CAFFE_CUDA_NUM_THREADS>>>(
      numsp_, feature, channels_);

    int num = 0, idx;
    const int di[4] = {-1, -width_, 1, width_};
    vector<vector<int> > adjacent(numsp_);
    for (int i = 0; i < numsp_; i++)
      adjacent[i].reserve(5);
    for (int r = 1; r < height_ - 1; r++) {
      for (int c = 1; c < width_ - 1; c++) {
        int curr = static_cast<int>(sp_data[idx = r*width_ + c]);
        for (int k = 0; k < 4; k++) {
          int pre = static_cast<int>(sp_data[idx + di[k]]);
        	if (curr > pre) {
        		vector<int>::iterator iter = std::find(adjacent[curr].begin(), adjacent[curr].end(), pre);
        		if (iter == adjacent[curr].end()) {
        			adjacent[curr].push_back(pre);
              num++;
        		}
          }
        }
      }
    }

    vector<int> dis_shape(4, 1);
    dis_shape[3] = num*3;
    Blob<Dtype> distanse;
    distanse.Reshape(dis_shape);
    Dtype* weight = distanse.mutable_cpu_data();

    idx = 0;
    for (int i = 0; i < numsp_; i++) {
      for (int j = 0; j < adjacent[i].size(); j++) {
        weight[idx++] = i;
        weight[idx++] = adjacent[i][j];
        idx++;
      }
    }
    CHECK_EQ(num, idx/3) << "Not equal";

    weight = distanse.mutable_gpu_data();
    DissimilarityForward<Dtype><<<CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS>>>(
        num, feature, weight, channels_);
    weight = distanse.mutable_cpu_data();
    // segment
    universe *u = segment_graph<Dtype>(numsp_, num, weight, bound);

    // post process small components
    for (int i = 0; i < num; i++) {
      int a = u->find(int(weight[i*3 + 0]));
      int b = u->find(int(weight[i*3 + 1]));
      if ((a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size)))
        u->join(a, b);
    }

    int* tables = new int[numsp_];
    for(int i = 0; i < numsp_; i++) {
      tables[i] = u->find(i);
    }
    for (int y = 0; y < height_; y++) {
      for (int x = 0; x < width_; x++) {
        top_data[y*width_ + x] = tables[static_cast<int>(sp_data[y*width_ + x])];
      }
    }
    delete u;
    delete [] tables;

    //finish = clock();
    //std::cout << (finish - start)*1.0 / CLOCKS_PER_SEC << " (s)" << std::endl;
}

template <typename Dtype>
void EgbSegmentLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

INSTANTIATE_LAYER_GPU_FUNCS(EgbSegmentLayer);

} //namespace caffe
