/*
 * HoughVotingScheme.cpp
 *
 *  Created on: Aug 8, 2016
 *      Author: munjalbharti
 */
#include "HoughVotingScheme.h"

using namespace cv ;

#include "ObjectDetectionUtils.h"



HoughVotingScheme::HoughVotingScheme(std::string dir,Vec2d bin, int threshold) {
	out_dir = dir;
	bin_size = bin;
	thresh = threshold;
}

HoughVotingScheme::~HoughVotingScheme() {}

cv::Mat get_hsv_image(cv::Mat img)
{
	cv::normalize(img, img, 0, 1., cv::NORM_MINMAX);
	cv::Mat hsv=cv::Mat(img.rows,img.cols,CV_8UC3, cvScalar(0,0,0));

	for (int i=0; i < img.rows;i++){
		for(int j=0;j< img.cols;j++){
			Vec3b pix;
			pix[0]=150*img.at<float>(i,j);
			pix[1]=255;
			pix[2]=255;
			hsv.at<Vec3b>(i,j)=pix;
		 }
		}

	cv::Mat bgr;
	cv::cvtColor(hsv, bgr,CV_HSV2BGR);

	return bgr ;

}


HoughVotingScheme::HoughVoting_Object_Centers_Out  HoughVotingScheme:: find_object_centers(Mat offset){


    int offset_rows = offset.rows;
	int offset_cols = offset.cols;
	int bin_size_x = bin_size[0];
	int bin_size_y = bin_size[1];


    int r_center_mask_rows=offset_rows/bin_size_y;
    int r_center_mask_cols=offset_cols/bin_size_x;

    Mat r_center_mask = Mat::zeros(r_center_mask_rows, r_center_mask_cols, CV_32FC1);

    std::map<int,std::vector<Point2d> > indexVsContrib;

    for (int y=0; y < offset_rows; y++){
	for(int x=0; x< offset_cols; x++){

	    int x1= round(x + offset.at<Vec2f>(y,x)[0]);
	    int y1= round(y + offset.at<Vec2f>(y,x)[1]);

	    if(x1 < 0 ){x1=0;}
	    if(x1 >= offset_cols){ x1=offset_cols-1;}
		if(y1 < 0){ y1=0;}
		if(y1 >= offset_rows){ y1=offset_rows-1;}

	    int c_m_y=floor(y1/bin_size_y);
	    int c_m_x=floor(x1/bin_size_x);

	    Point2d contr_pixel(x,y);

	    float val=r_center_mask.at<float>(c_m_y,c_m_x);
	    int ind=get_serial_index(offset_cols,c_m_y,c_m_x);
	    if(val == 0){
	    	  r_center_mask.at<float>(c_m_y,c_m_x)=1;
	    	  std::vector<Point2d> v;
	    	  v.push_back(contr_pixel);
	    	  indexVsContrib[ind]= v;
	    }else {
	    	  r_center_mask.at<float>(c_m_y,c_m_x)= val+1;
	    	  std::vector<Point2d> v = indexVsContrib[offset_cols * c_m_y + c_m_x];
	    	  v.push_back(contr_pixel);
	    	  indexVsContrib[ind]= v;
	    }
	}

}



     std::string ss= out_dir + "orig_heat_map.txt" ;
     save_mask(r_center_mask,ss);


     std::vector<Point2d> centers = post_process_heat_map(r_center_mask);

     std::map<int,std::vector<Point2d> > indexVsContribProcessed;

     for ( int i=0 ; i < (signed int)centers.size(); i++){
    	Point2d c= centers.at(i);
    	int ind=get_serial_index(offset_cols,c.y,c.x);
    	indexVsContribProcessed[ind]=indexVsContrib[ind];
     }

     HoughVoting_Object_Centers_Out out ;
     out.centers=centers;
     out.indexVsContrib=indexVsContrib;

     return out ;


}

std::vector<Point2d> HoughVotingScheme::post_process_heat_map(Mat heat_map){


	Mat r_normalised;
	cv::normalize(heat_map, r_normalised, 0, 1, NORM_MINMAX, -1);

	//std::string ss1=out_dir + "norm_heat_map.txt" ;
    //save_mask(r_normalised,ss1);

	//Thresholding
	Mat thresholded_map;
	float max_val=1;
	float thresh=0.1;
	threshold(r_normalised, thresholded_map, thresh, max_val,THRESH_BINARY);
	//non-maxima suppression


	std::vector<Point2d> centers;

	for (int m=0 ; m < thresholded_map.rows;m++){
		for(int n=0; n < thresholded_map.cols; n++){
			float val=thresholded_map.at<float>(m,n);
			if(val == max_val){
				Point2d p=Point(n,m);
				centers.push_back(p);
			}
		}
	}

	  std::string filename = out_dir +"thresh_heat_map.txt" ;
	  save_mask(thresholded_map,filename);




	return centers;
}


bool HoughVotingScheme::save_heat_map(Mat r_center_mask, std::string filename){

	Mat r_normalised;
    cv::normalize(r_center_mask, r_normalised, 0, 255, NORM_MINMAX, CV_8UC1);


	Mat im_color;
	applyColorMap(r_normalised, im_color, COLORMAP_JET);

	std::string file=filename.substr (0,filename.length()-4);

	//char win_name[50];
//	sprintf(win_name, file.c);
	cv::namedWindow(file.c_str(),cv::WINDOW_NORMAL);

	imshow(file.c_str(), im_color);



	std::string ss=file + ".png" ;

	imwrite(ss, im_color);

	return true;
}

bool HoughVotingScheme::save_mask(Mat r_center_mask, std::string filename){
	int r_center_mask_rows= r_center_mask.rows ;
	int r_center_mask_cols= r_center_mask.cols ;


	std::ofstream output(filename.c_str());
	if (!output.is_open()){
		  std::cout << "could not open file" << std::endl ;
		  return false;
	}

	for (int i=0; i < r_center_mask_rows;i++){
		for(int j=0; j < r_center_mask_cols;j++){
			float val=r_center_mask.at<float>(i,j);
			output << val << " " ;
		     output << "\n" ;
	}
	}

	output.close();

	save_heat_map(r_center_mask,filename);


	return true;
}



