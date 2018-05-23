#ifndef KMEAN_SEGMENT_
#define KMEAN_SEGMENT_

#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "caffe/segment/image.h"

typedef unsigned int UINT;
using namespace cv;
using namespace std;

class Kmeanseg {
public:
  Kmeanseg();
  virtual ~Kmeanseg();

  image<rgb>* PerformKmeans_ForGivenK(image<feature>* im, int K);

private:
  void normalize(image<feature>* im);

  void PerformSegmentation_UsingMetric(image<feature>* im, vector<feature>& kseedsf, vector<double>& kseedsx,
    vector<double>& kseedsy, int* klabels, const int& STEP, const int& NUMITR);

  void GetSeeds_ForGivenK(image<feature>* im, vector<feature>& kseedsf, vector<double>& kseedsx,
    vector<double>& kseedsy, int K, bool perturbseeds, vector<double>& edgemag);

  void EnforceLabelConnectivity(const int* labels, int* nlabels, int& numlabels, const int& K);

  void PerturbSeeds(image<feature>* im, vector<feature>& kseedsf, vector<double>& kseedsx,
    vector<double>& kseedsy, const vector<double>& edges);

  void DetectImgEdges(image<feature>* im, vector<double>& edges);

  double ComputeDistance(feature& vecneigh, feature& vecseed);

  void AdaptiveGetSeeds_ForGivenK(image<feature>* im, vector<feature>& kseedsf, vector<double>& kseedsx,
	  vector<double>& kseedsy, int K, bool perturbseeds, vector<double>& edgemag);
  rgb random_rgb();
private:
  int* labels;
  int m_width;
  int m_height;
  int m_channels;
  int sz;
};


#endif
