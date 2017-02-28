/*
 * HoughVotingScheme.h
 *
 *  Created on: Aug 8, 2016
 *      Author: munjalbharti
 */

#include <Eigen/Dense>

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <iomanip>
#include <map>

#include <math.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//#include "ObjectDetectionUtils.h"

#ifndef HOUGHVOTINGSCHEME_H_
#define HOUGHVOTINGSCHEME_H_

using namespace cv ;

class HoughVotingScheme {
public:
	HoughVotingScheme();
	HoughVotingScheme(std::string out_dir,Vec2d bin_size, int thresh);

	virtual ~HoughVotingScheme();

	struct HoughVoting_Object_Centers_Out{
		 std::vector<Point2d> centers ;
		 std::map<int,std::vector<Point2d> > indexVsContrib;
	 };

	HoughVoting_Object_Centers_Out find_object_centers(Mat offset);


private :

	std::string out_dir ;
	Vec2d bin_size;
	int thresh;

	std::vector<Point2d> post_process_heat_map(Mat heat_map);
	bool save_mask(Mat r_center_mask, std::string filename);
	bool save_heat_map(Mat r_center_mask, std::string filename);
};




#endif /* HOUGHVOTINGSCHEME_H_ */
