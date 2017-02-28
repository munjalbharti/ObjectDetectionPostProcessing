/*
 * MeanShift.h
 *
 *  Created on: Aug 14, 2016
 *      Author: munjalbharti
 */

#ifndef MEANSHIFT_H_
#define MEANSHIFT_H_

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



class MeanShift {
public:
	MeanShift();
	MeanShift(std::string out_dir);

	virtual ~MeanShift();

	struct MeanShift_Object_Centers_Out{
		std::vector<cv::Point2d> centers;
		std::map<int,std::vector<cv::Point2d>> indexVsContrib;
	};

	MeanShift_Object_Centers_Out find_object_centers(cv::Mat offset);

private :

	std::string out_dir ;

	std::vector<Eigen::Vector2f> clusterTranslation(std::vector<Eigen::Vector2f> &points);
	bool save_centers(const std::vector<Eigen::Vector2f>  centers , const std::string filename);
	bool read_centers( const std::string file_name,  std::vector<Eigen::Vector2f>  &centers );
	bool save_indxVsContrib(const  std::map<int,int> indexVsContrib, const std::string filename);



};

#endif /* MEANSHIFT_H_ */
