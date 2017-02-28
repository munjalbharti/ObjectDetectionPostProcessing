#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <iomanip>
#include <map>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "ObjectDetectionUtils.h"

#include "HoughVotingScheme.h"
#include "MeanShift.h"


#ifndef WIN64
    #define EIGEN_DONT_ALIGN_STATICALLY
#endif

using namespace cv ;
using namespace Eigen ;

///Set parameters here.. TODO: take from commandline

int offst_rows=256;
int offst_cols=256;
std::string out_dir;
std::string in_dir;
int method=1; //options: 1: hough Method, 2: meanShift

bool read_offset(const std::string offset_file_name, Mat &offset);
void display_bounding_box(Mat img, std::vector<Point2d> centers, std::map <int,std::vector<Point2d> > indexVsContrib);

bool save_offset(const Mat offset, const std::string filename);
std::vector<Rect> find_bounding_boxes(std::vector<Point2d> centers, std::map<int,std::vector<Point2d> > indexVsContrib);


int main(int argc, char *argv[])
{

	std::string base_dir ;

	if(argc != 2){
		base_dir="/home/munjalbharti/objectdetection/data/";
		std::cout << "data directory not passed..using default : " <<  base_dir << std::endl ;
	}else{
		base_dir=std::string(argv[1]);
		std::cout << "using data directory : " << base_dir << std::endl ;
	}

	out_dir= base_dir + "/out/";
	in_dir= base_dir + "/in/" ;


	std::string ss_offset= in_dir + "2007_000129_offset.txt" ;

	Mat offset = Mat::zeros(offst_rows, offst_cols, CV_32FC2);
	if(!read_offset(ss_offset, offset)){
		return false ;
	}


//	 std::string ss=out_dir + "offset_verify.txt" ;
//    if (!save_offset(offset, ss.str())){
//    	return false;
//    }


	std::vector<cv::Point2d> centers;
	std::map<int,std::vector<cv::Point2d>> indexVsContrib;
	Mat img=Mat::zeros(offst_rows, offst_cols, CV_8UC3)	;

	if(method==1){
		int thresh=0;
		Vec2d bin_size(1,1);
		HoughVotingScheme houghVotingScheme(out_dir,bin_size,thresh);
		HoughVotingScheme::HoughVoting_Object_Centers_Out out=houghVotingScheme.find_object_centers(offset);

		display_bounding_box(img,out.centers,out.indexVsContrib);
	}else {
		MeanShift meanShift(out_dir);
		MeanShift::MeanShift_Object_Centers_Out out= meanShift.find_object_centers(offset);

		display_bounding_box(img,out.centers,out.indexVsContrib);
	}

	cv::waitKey(0);
    return 0;
}





void display_bounding_box(Mat img, std::vector<Point2d> centers, std::map <int,std::vector<Point2d> > indexVsContrib){
	std::vector<Rect> bb=find_bounding_boxes(centers, indexVsContrib);

	char win_name[50];
	sprintf(win_name, "Output");
	int no_of_centers=centers.size();
	for(int k=0; k < no_of_centers;k++){
		Point2d p = centers.at(k);
		Point2d center(p.x,p.y);

		int c =get_serial_index(offst_cols, p.y,p.x);

		std::vector<Point2d> contri =  indexVsContrib[c];
		int count= (signed int) contri.size();

		for (int l=0 ; l < count; l++){
			       Point2d p=contri.at(l);

			       Vec3b pix;
			       pix[0]=0;
			       pix[1]=255;
			       pix[2]=0;
			      // img.at<Vec3b>(p.y,p.x)=pix;
			       circle( img,p,1,Scalar( 255, 255, 255 ),-1,8 );

		}

		circle( img,center,5,Scalar( 0, 0, 255 ),-1,8 );

		//draw  bounding boxes
		while(!bb.empty()){
			Rect rect = bb.back();
			bb.pop_back();
			rectangle(img, rect, Scalar(255,0,0), 2,8);
		}
	}

	cv::namedWindow(win_name,cv::WINDOW_NORMAL);
    imshow(win_name, img);

    std::stringstream ss;
    ss << out_dir << "bounding_box.png" ;
    imwrite(ss.str(), img );


}


bool save_offset(const Mat offset, const std::string filename){

	std::ofstream output1(filename.c_str());
	if (!output1.is_open()){
		   std::cout << "could not open file" << filename << std::endl ;
		   return false;
	}

	for (int i=0; i < offst_rows;i++){
	  for(int j=0; j < offst_cols;j++){
			float val1=offset.at<Vec2f>(i,j)[0];
			float val2=offset.at<Vec2f>(i,j)[1];
			output1 << val1 << " " << val2 << "\n";
		}
	}
	output1.close();
	return true;
}

bool read_offset( const std::string offset_file_name, Mat &offset){

	    std::ifstream offset_file(offset_file_name.c_str());
	    if (!offset_file.is_open())
	     {
	    	 std::cout << "Cannot open file " << offset_file_name <<  std::endl ;
	    	 return false;
	     }
	    std::string off;
	    int count=1;
	    while (std::getline(offset_file, off))
	    	{
	    	    std::istringstream is(off);
	    	    	//std::cout << off << std::endl ;
	    	    Vec2f pix;
	    	    int i=0;
	    	    float n;
	    	    while( is >> n ) {
	    	      pix[i]=n;
	    	      i=i+1;
	    	    }

	    	   	int row=ceil(count/offst_cols)-1;
	    	   	int col= (count-offst_cols*row)-1 ;

	   	       offset.at<Vec2f>(row,col)=pix;
	           count++;

	    	    }

	    offset_file.close();

return true;
}



std::vector<Rect> find_bounding_boxes(std::vector<Point2d> centers, std::map<int,std::vector<Point2d> > indexVsContrib){


	std::vector<Rect> bounding_boxes;
	while(!centers.empty()){
			Point2d p = centers.back();
			centers.pop_back();

			Point center(p.x,p.y);
			int c =get_serial_index(offst_cols, p.y,p.x);

	        std::vector<Point2d> contri =  indexVsContrib[c];
			int count= (signed int) contri.size();

			int min_x=1000, min_y= 1000, max_x=0, max_y=0;


			for (int l=0 ; l < count; l++){
				       Point2d p=contri.at(l);


				       if(min_x > p.x ){
				    	   min_x=p.x;
				       }


				       if(min_y > p.y){
				    	   min_y=p.y;
				       }

				       if(max_x < p.x ){
				    	   max_x=p.x;
				        }

				       if(max_y < p.y){
				      	   max_y=p.y;
				      	}


			}

			Rect tl(Point(min_x-1,min_y-1), Point(max_x+1,max_y+1));
		    bounding_boxes.push_back(tl);
	}

	return bounding_boxes;
}

//void display_cluster(Mat img,  std::vector<Point2d> centers,std::map <int,std::vector<Point2d> > indexVsContrib ){
//    	char win_name[50];
//		sprintf(win_name, "Cluster");
//		int no_of_centers=centers.size();
//		for(int k=0; k < no_of_centers;k++){
//			Point2d center = centers.at(k);
//			//Point2d center(p(0),p(1));
//			int x=center.x;
//			int y=center.y;
//			int c =get_serial_index(offst_cols, y,x);
//
//			std::vector<Point2d> contri =  indexVsContrib[c];
//			int count= (signed int) contri.size();
//
//			for (int l=0 ; l < count; l++){
//						Point2d p=contri.at(l);
//						Vec3b pix;
//						pix[0]=0;
//						pix[1]=255;
//						pix[2]=0;
//						// img.at<Vec3b>(p.y,p.x)=pix;
//						circle( img,p,2,Scalar( 255, 255, 255 ),-1,8 );
//
//					}
//
//
//			circle( img,center,	3,Scalar( 0, 0, 255 ),-1,8 );
//		}
//
//		cv::namedWindow(win_name,cv::WINDOW_NORMAL);
//	    imshow(win_name, img);
//
//
//}




//struct obj_rects{
//	int x_mins[];
//	int y_mins[];
//	int widths[];
//	int heights[];
//
//}rects_obj;

	//threshold(r_center_mask, r_center_mask, 2.0, 1., CV_THRESH_BINARY);

	//char win_nam[50];
	//sprintf(win_nam, "HeatMaptHRH");
	//cv::namedWindow(win_nam,cv::WINDOW_AUTOSIZE);
	//imshow(win_nam, r_center_mask);
	//cv::waitKey(0);

//      std::ifstream rect_file("/home/munjalbharti/2007_000129_rects.txt");
//      if (rect_file.is_open())
//      {
//		  std::string str;
//		  int line_no=1;
//
//			  while (std::getline(rect_file, str))
//			  {
//
//				std::istringstream stream(str);
//				int obj_no=0;
//				int n;
//
//				while(stream >> n){
//				  switch(line_no){
//					  case 1:
//						  rects_obj.x_mins[obj_no]=n;
//						  break;
//					  case 2:
//						  rects_obj.y_mins[obj_no]=n;
//						  break;
//
//					  case 3:
//						  rects_obj.widths[obj_no]=n;
//						  break;
//
//					  case 4:
//						 rects_obj.heights[obj_no]=n;
//						  break;
//					  }
//				  obj_no=obj_no+1;
//				}
//
//				line_no++;
//      }
//
//      }

