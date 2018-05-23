#include <cfloat>
#include <cmath>
#include <iostream>
#include <fstream>
#include <cstdlib>
//#include <random>
#include "caffe/segment/kmeans.hpp"

using namespace std;
using namespace cv;

Kmeanseg::Kmeanseg() {
  labels = NULL;
}
Kmeanseg::~Kmeanseg() {
  if(labels)
    delete[] labels;
}
void Kmeanseg::normalize(image<feature>* im) {
	double max_value = -DBL_MAX, min_value = -max_value;
	for (int i = 0; i < m_height; i++) {
		for (int j = 0; j < m_width; j++) {
			double current_min_value = *min_element(imRef(im, j, i).data, imRef(im, j, i).data + 64);
			double current_max_value = *max_element(imRef(im, j, i).data, imRef(im, j, i).data + 64);
			if (min_value > current_min_value) {
				min_value = current_min_value;
			}
			if (max_value < current_max_value) {
				max_value = current_max_value;
			}
		}
	}
	double diff = max_value - min_value;
	for (int i = 0; i < m_height; i++) {
		for (int j = 0; j < m_width; j++) {
			for (int c = 0; c < C_DIMS; c++) {
				imRef(im, j, i).data[c] = (imRef(im, j, i).data[c] - min_value) / diff;
			}
		}
	}
}
image<rgb>* Kmeanseg::PerformKmeans_ForGivenK(image<feature>* im, int K) {

  vector<double> kseedsx(0);
  vector<double> kseedsy(0);
  vector<feature> kseedsf(0);

  m_width = im->width();
  m_height = im->height();
  m_channels = C_DIMS;
  sz = m_width * m_height;
  int NUMITR = 30;

  labels = new int[sz];
  memset(labels, -1, sizeof(int)*sz);
  bool perturbseeds(true);
  vector<double> edgemag(0);

  //normalize the input data
  //normalize(im);
  if(perturbseeds) DetectImgEdges(im, edgemag);
  //GetSeeds_ForGivenK(im, kseedsf, kseedsx, kseedsy, K, perturbseeds, edgemag);
  AdaptiveGetSeeds_ForGivenK(im, kseedsf, kseedsx, kseedsy, K, perturbseeds, edgemag);
  int STEP = sqrt(double(sz)/double(K)) + 2.0;//adding a small value in the even the STEP size is too small.
  PerformSegmentation_UsingMetric(im, kseedsf, kseedsx, kseedsy, labels, STEP, NUMITR);

  //int numlabels = kseedsf.size();
  //int* nlabels = new int[sz];
  //EnforceLabelConnectivity(labels, nlabels, numlabels, K);
  //for(int i = 0; i < sz; i++ ) labels[i] = nlabels[i];
  //if(nlabels) delete [] nlabels;

  rgb* color = new rgb[sz];
  for(int i = 0; i < sz; i++) color[i] = random_rgb();
  //Mat segmentImg(m_height, m_width, CV_8UC3);
  image<rgb>* im_seg = new image<rgb>(m_width, m_height);
  for(int y = 0; y < m_height; y++) {
    for(int x = 0; x < m_width; x++) {
      int pos = y * m_width + x;
      imRef(im_seg, x, y) = color[labels[pos]];
    }
  }
  delete[] color;
  return im_seg;
}

void Kmeanseg::PerformSegmentation_UsingMetric(image<feature>* im, vector<feature>& kseedsf, vector<double>& kseedsx,
   vector<double>& kseedsy, int* klabels, const int&	STEP, const int& NUMITR) {

  const int numk = kseedsf.size();
  //----------------
  int offset = STEP;
  if(STEP < 10) offset = STEP*1.5;
  offset = offset*3;
  //----------------

  feature init_value;
  for(int i = 0; i < m_channels; i++)  init_value.data[i] = 0;

  vector<feature> sigmaf(numk, init_value);
  vector<double> sigmax(numk, 0);
  vector<double> sigmay(numk, 0);
  vector<int>  clustersize(numk, 0);
  vector<double> inv(numk, 0);
  vector<double> distfeat(sz, DBL_MAX);
  vector<double> distxy(sz, DBL_MAX);
  vector<double> distvec(sz, DBL_MAX);
  vector<double> maxxy(numk, STEP*STEP);
  vector<double> maxfeat(numk, 10 * 10);

  double M = 0.1;
  double invwt = 1.0 / ((offset / M)*(offset / M));

  double oldVar = -1, newVar = 0;

  int numitr(0);
  while(numitr < NUMITR && fabs(newVar - oldVar) > 0.01) {
    numitr++;
    distvec.assign(sz, DBL_MAX);

    for(int n = 0; n < numk; n++) {
      int y1 = std::max(0,		(int)(kseedsy[n]-offset));
      int y2 = std::min(m_height,	(int)(kseedsy[n]+offset));
      int x1 = std::max(0,		(int)(kseedsx[n]-offset));
      int x2 = std::min(m_width,	(int)(kseedsx[n]+offset));

      for(int y = y1; y < y2; y++) {
        for(int x = x1; x < x2; x++) {

          int i = y*m_width + x;
          if(y >= m_height || x >= m_width || y < 0 || x < 0) {
            cout << "Error position out of image range......" << endl;
            return;
          }
          //dist_feature[i] = ComputeDistance(imRef(im, x, y), kseedsf[n]);
           distxy[i] = (x - kseedsx[n])*(x - kseedsx[n]) + (y - kseedsy[n])*(y - kseedsy[n]);
          //------------------------------------------------------------------------
          //double dist = distlab[i]/maxlab[n] + distxy[i]*invxywt;//only varying m, prettier superpixels
		   distfeat[i] = ComputeDistance(imRef(im, x, y), kseedsf[n]); //+ distxy * invwt;
          //------------------------------------------------------------------------
		  double dist = distxy[i] * invwt + distfeat[i];

		  if(dist < distvec[i]) {
            distvec[i] = dist;
            klabels[i]  = n;
          }
        }
      }
    }

  	oldVar = newVar;
  	newVar = 0;

    sigmax.assign(numk, 0);
    sigmay.assign(numk, 0);
    sigmaf.assign(numk, init_value);
  	clustersize.assign(numk, 0);

    for(int i = 0; i < m_height; i++) {
      for(int j = 0; j < m_width; j++) {
        int index = i * m_width + j;
        if(klabels[index] < 0) {
  			cout << "Error ..." << endl;
  			return;
  		}
        feature pf = imRef(im, j, i);
        sigmax[klabels[index]] += j;
  		sigmay[klabels[index]] += i;
        for(int c = 0; c < m_channels; c++)
          sigmaf[klabels[index]].data[c] += pf.data[c];

        clustersize[klabels[index]]++;
		    newVar += ComputeDistance(pf, kseedsf[klabels[index]]);
      }
    }
    for(int k = 0; k < numk; k++) {
      if(clustersize[k] <= 0) clustersize[k] = 1;
      inv[k] = 1.0/double(clustersize[k]);
	}

    for(int k = 0; k < numk; k++) {
	  kseedsx[k] = sigmax[k]*inv[k];
	  kseedsy[k] = sigmay[k]*inv[k];
      for(int c = 0; c < m_channels; c++)
        kseedsf[k].data[c] = sigmaf[k].data[c]*inv[k];
	  //cout << "seed number: " << numitr << endl;
	 // cout << "X position: " << kseedsx[k] << "Y position: " << kseedsy[k] << endl;
	}
	//cout << "========================" << endl;
  }

}

void Kmeanseg::GetSeeds_ForGivenK(image<feature>* im, vector<feature>& kseedsf, vector<double>& kseedsx,
  vector<double>& kseedsy, int K, bool perturbseeds, vector<double>& edgemag) {

	double step = sqrt(double(sz)/double(K));
	int xoff = step/2;
	int yoff = step/2;

	int n(0);int r(0);
	for(int y = 0; y < m_height; y++) {
		int Y = y*step + yoff;
		if(Y > m_height-1) break;

		for(int x = 0; x < m_width; x++) {
			//int X = x*step + xoff;//square grid
			int X = x*step + (xoff<<(r&0x1));//hex grid
			if(X > m_width-1) break;

			kseedsx.push_back(X);
			kseedsy.push_back(Y);
			kseedsf.push_back(imRef(im, X, Y));
			//cout << "X position: " << X << "Y position: " << Y << endl;
			n++;
		}
		r++;
	}

	if(perturbseeds) {
		PerturbSeeds(im, kseedsf, kseedsx, kseedsy, edgemag);
	}
}

void Kmeanseg::AdaptiveGetSeeds_ForGivenK(image<feature>* im, vector<feature>& kseedsf, vector<double>& kseedsx,
	vector<double>& kseedsy, int K, bool perturbseeds, vector<double>& edgemag) {
	//first random seed
	srand(time(NULL));
	int h = rand() % m_height;
	int w = rand() % m_width;
	if (h >= m_height || w >= m_width || h < 0 || w < 0) {
		cout << "Error position out of image range......" << endl;
		return;
	}

	kseedsy.push_back(h);
	kseedsx.push_back(w);
	kseedsf.push_back(imRef(im, w, h));

	vector<double> distvec(sz, DBL_MAX);
	vector<double> prob(sz, 0);

	//compute distance from seleted seeds
	int numk = kseedsf.size();
	while (numk < K) {
		distvec.assign(sz, DBL_MAX);
		prob.assign(sz, 0);
		double sum = 0;

		for (int y = 0; y < m_height; y++) {
			for (int x = 0; x < m_width; x++) {
				int i = y * m_width + x;
				for (int n = 0; n < numk; n++) {
					double dist = ComputeDistance(imRef(im, x, y), kseedsf[n]);
					if (dist < distvec[i]) {
						distvec[i] = dist;
					}
				}
			}
		}

		for (int i = 0; i < sz; i++) {
			sum += distvec[i] * distvec[i];
		}
		for (int i = 0; i < sz; i++) {
			prob[i] = distvec[i] * distvec[i] / sum;
		}
		//generate random number range from 0 to 1
		double random_p = ((double)rand()) / RAND_MAX;
		//std::default_random_engine random(time(NULL));
		//std::uniform_real_distribution<double> dist(0.0, 1.0);

		//double random_p = dist(random);
		//cout << "random probablity:" << random_p << endl;

		double sumprob = 0;
		int i;
		for (i = 0; i < sz; i++) {
			sumprob += prob[i];
			if (random_p < sumprob) break;
		}
		//selete another seed point
		kseedsx.push_back(i % m_width);
		kseedsy.push_back(i / m_width);
		kseedsf.push_back(imRef(im, i % m_width, i / m_width));
		//cout << "X position: " << i % m_width << "Y position: " << i / m_width << endl;
		numk = kseedsf.size();
	}
	if (perturbseeds) {
		PerturbSeeds(im, kseedsf, kseedsx, kseedsy, edgemag);
	}
}
void Kmeanseg::DetectImgEdges(image<feature>* im, vector<double>& edges) {

  edges.resize(sz, 0);
  for( int j = 1; j < m_height-1; j++ ) {
    for( int k = 1; k < m_width-1; k++ ) {
      feature x1 = imRef(im, k+1, j), x2 = imRef(im, k-1, j);
      feature y1 = imRef(im, k, j-1), y2 = imRef(im, k, j+1);
      double dx = 0, dy = 0;
      for(int i = 0; i < m_channels; i++) {
        dx += (x1.data[i] - x2.data[i]) * (x1.data[i] - x2.data[i]);
        dy += (y1.data[i] - y2.data[i]) * (y1.data[i] - y2.data[i]);
      }
      //edges[i] = (sqrt(dx) + sqrt(dy));
      int pos = j*m_width + k;
      edges[pos] = (dx + dy);

    }
  }
  Mat edgemap(m_height, m_width, CV_8UC1);
  for (int j = 0; j < m_height; j++) {
	  for (int k = 0; k < m_width; k++) {
		  int pos = j*m_width + k;
		  edgemap.at<uchar>(j, k) = edges[pos] * 10000;
	  }
  }
  imwrite("data/edgemap.png", edgemap);
}

void Kmeanseg::PerturbSeeds(image<feature>* im, vector<feature>& kseedsf, vector<double>& kseedsx,
  vector<double>& kseedsy, const vector<double>& edges) {

  const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

	int numseeds = kseedsf.size();

	for(int n = 0; n < numseeds; n++) {
		int ox = kseedsx[n];//original x
		int oy = kseedsy[n];//original y
		int oind = oy*m_width + ox;

		int storeind = oind;
		for(int i = 0; i < 8; i++) {
			int nx = ox+dx8[i];//new x
			int ny = oy+dy8[i];//new y

			if( nx >= 0 && nx < m_width && ny >= 0 && ny < m_height) {
				int nind = ny*m_width + nx;
				if(edges[nind] < edges[storeind]) {
					storeind = nind;
				}
			}
		}
		if(storeind != oind) {
			kseedsx[n] = storeind%m_width;
			kseedsy[n] = storeind/m_width;
      kseedsf[n] = imRef(im, storeind%m_width, storeind/m_width);
		}
	}
}
double Kmeanseg::ComputeDistance(feature& v1, feature& v2) {
  double dist = 0;
  for(int i = 0; i < m_channels; i++) {
    dist += (v1.data[i] - v2.data[i]) * (v1.data[i] - v2.data[i]);
	//dist += fabs(v1.data[i] - v2.data[i]);
  }
  return dist;
}
void Kmeanseg::EnforceLabelConnectivity(const int* labels, int* nlabels,
  int& numlabels, const int& K) {
  const int dx4[4] = {-1,  0,  1,  0};
	const int dy4[4] = { 0, -1,  0,  1};

	const int SUPSZ = sz/K;
	//nlabels.resize(sz, -1);
	for(int i = 0; i < sz; i++) nlabels[i] = -1;
	int label(0);
	int* xvec = new int[sz];
	int* yvec = new int[sz];
	int oindex(0);
	int adjlabel(0);//adjacent label
	for(int j = 0; j < m_height; j++) {
		for(int k = 0; k < m_width; k++) {
			if(0 > nlabels[oindex]) {
				nlabels[oindex] = label;
				//--------------------
				// Start a new segment
				//--------------------
				xvec[0] = k;
				yvec[0] = j;
				//-------------------------------------------------------
				// Quickly find an adjacent label for use later if needed
				//-------------------------------------------------------
				for(int n = 0; n < 4; n++) {
					int x = xvec[0] + dx4[n];
					int y = yvec[0] + dy4[n];
					if((x >= 0 && x < m_width) && (y >= 0 && y < m_height)) {
						int nindex = y*m_width + x;
						if(nlabels[nindex] >= 0) adjlabel = nlabels[nindex];
					}
				}

				int count(1);
				for(int c = 0; c < count; c++) {
					for(int n = 0; n < 4; n++) {
						int x = xvec[c] + dx4[n];
						int y = yvec[c] + dy4[n];

						if((x >= 0 && x < m_width) && (y >= 0 && y < m_height)) {
							int nindex = y*m_width + x;

							if(0 > nlabels[nindex] && labels[oindex] == labels[nindex]) {
								xvec[count] = x;
								yvec[count] = y;
								nlabels[nindex] = label;
								count++;
							}
						}
					}
				}
				//-------------------------------------------------------
				// If segment size is less then a limit, assign an
				// adjacent label found before, and decrement label count.
				//-------------------------------------------------------
				if(count <= SUPSZ >> 2) {
					for(int c = 0; c < count; c++) {
						int ind = yvec[c]*m_width+xvec[c];
						nlabels[ind] = adjlabel;
					}
					label--;
				}
				label++;
			}
			oindex++;
		}
	}
	numlabels = label;

	if(xvec) delete [] xvec;
	if(yvec) delete [] yvec;

}
rgb Kmeanseg::random_rgb() {
  rgb c;
  c.r = (uchar)rand()%256;
  c.g = (uchar)rand()%256;
  c.b = (uchar)rand()%256;
  return c;
}
