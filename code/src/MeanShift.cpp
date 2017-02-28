/*
 * MeanShift.cpp
 *
 *  Created on: Aug 14, 2016
 *      Author: munjalbharti
 */

#include "MeanShift.h"
#include "ObjectDetectionUtils.h"


MeanShift::MeanShift() {
	// TODO Auto-generated constructor stub

}

MeanShift::MeanShift(std::string dir) {
	out_dir = dir;
}


MeanShift::~MeanShift() {
	// TODO Auto-generated destructor stub
}

using namespace std ;
using namespace cv ;
using namespace Eigen;



MeanShift::MeanShift_Object_Centers_Out MeanShift::find_object_centers(cv::Mat offset){
	vector<Vector2f> points;
	vector<Point2d> orig_points;


    int offset_rows = offset.rows;
	int offset_cols = offset.cols;


    for (int y=0; y < offset_rows; y++){
	for(int x=0; x< offset_cols; x++){

	    int x1= round(x + offset.at<Vec2f>(y,x)[0]);
	    int y1= round(y + offset.at<Vec2f>(y,x)[1]);

	    if(x1 < 0 ){x1=0;}
	    if(x1 >= offset_cols){ x1=offset_cols-1;}
		if(y1 < 0){ y1=0;}
		if(y1 >= offset_rows){ y1=offset_rows-1;}

		Vector2f vect(x1,y1);
		points.push_back(vect);

		Point2d orig_vect(x,y);
		orig_points.push_back(orig_vect);
	}

}

    std::vector<Vector2f>  centers = MeanShift::clusterTranslation(points);

//    std::vector<Vector2f>  centers  ;
//    std::string ss = out_dir + "centers.txt" ;
//    read_centers(ss,centers);

	std::map<int,std::vector<Point2d>> indexVsContrib;
    std::map<int,int> indexVsCount;

    int no_of_points= (signed int)points.size();

    std::vector<Point2d>  uniq_centers;

    for (int i=0; i < no_of_points; i++){
    	Vector2f  vec= centers.at(i);
    	Point2d  contr_pixel=orig_points.at(i);

    	float x=vec(0);
    	float y=vec(1);

    	int c_m_x = floor(x);
    	int c_m_y = floor(y);

        int ind=get_serial_index(offset_cols,c_m_y,c_m_x);
        int c=  indexVsContrib.count(ind);
        std::vector<Point2d> v = indexVsContrib[ind];

    	if(c == 0){
    		Point2d p(x,y);
    		uniq_centers.push_back(p);
    		indexVsCount[ind]=1;
    	}else {
    		indexVsCount[ind]= indexVsCount[ind]+1;
    	}

        v.push_back(contr_pixel);
    	indexVsContrib[ind]= v ;

  }


//    save_centers(centers,ss.str());
//    std::string ss1= out_dir + "indxVsCount.txt" ;
//    save_indxVsContrib(indexVsCount,ss1);


    std::vector<Point2d>  selected_centers;
    std::map<int,std::vector<Point2d>> selectedIndexVsContrib;

    int thresh=10;
    int unq_centers_count= (signed int)uniq_centers.size();


    for (int i=0; i < unq_centers_count;i++){
    	Point2d  vec= uniq_centers.at(i);
    	int ind=get_serial_index(offset_cols,vec.y, vec.x);
    	int count=  indexVsCount[ind];
    	if(count > thresh){
    		selectedIndexVsContrib[ind]=indexVsContrib[ind];
    		selected_centers.push_back(vec);
    	}

    }


    MeanShift_Object_Centers_Out out;
    out.centers=selected_centers;
    out.indexVsContrib=selectedIndexVsContrib;

    return out ;

}

std::vector<Vector2f> MeanShift::clusterTranslation(vector<Vector2f> &points)
{

    float bandwidth = 2;
    float eps = pow(0.001,2);

    auto computeMeanFlatKernel = [](Vector2f &center, vector<Vector2f> points, float dist)
    {
            float trans_error = dist*dist;
            Vector2f mean(0,0);
            int counter = 0;
            for (Vector2f &p : points)
            {
            	float n=(p-center).squaredNorm();
                if ( n > trans_error) continue;
                counter++;
                mean += p;
            }
            mean /= (float) counter;
            return mean;
    };

    vector<Vector2f> new_set = points;

    bool converged = false;

    vector<bool> processed(points.size(),false);

    int max_iterations = 1;
    for(int run=0;!converged && run < max_iterations; run++)
    {
        converged = true;
        for(size_t i=0; i < new_set.size(); ++i)
        {
            if (processed[i]) continue;
            Vector2f mean = computeMeanFlatKernel(new_set[i],points,bandwidth);
            processed[i] = (new_set[i]-mean).squaredNorm() < eps;
            new_set[i] = mean;
            converged &= processed[i];
        }
    }

    return new_set;
}

bool read_centers( const std::string file_name,  std::vector<Vector2f>  &centers ){

	    std::ifstream offset_file(file_name.c_str());
	    if (!offset_file.is_open())
	     {
	    	 std::cout << "Cannot open file " <<  file_name << std::endl ;
	    	 return false;
	     }
	    std::string off;
	    int count=1;
	    while (std::getline(offset_file, off))
	    	{
	    	    std::istringstream is(off);
	    	    	//std::cout << off << std::endl ;
	    	    Vector2f pix;
	    	    int i=0;
	    	    float n;
	    	    while( is >> n ) {
	    	      pix(i)=n;
	    	      i=i+1;
	    	    }

	    	   	 count++;
	    	   	 centers.push_back(pix);
	    	    }

	    offset_file.close();

return true;
}

bool save_centers(const std::vector<Vector2f>  centers , const std::string filename){
    std::ofstream output1(filename.c_str());
    if (!output1.is_open()){
               std::cout << "could not open file " <<  filename << std::endl ;
               return false;
    }

    int no_of_points=centers.size();

    for (int i=0; i < no_of_points; i++){
            Vector2f  vec= centers.at(i);
            float val1=vec[0];
            float val2=vec[1];
            output1 << val1 << "  " << val2 << "\n";

    }
    output1.close();
    return true;
}

bool save_indxVsContrib(const  std::map<int,int> indexVsContrib, const std::string filename){


	std::ofstream output1(filename.c_str());
    if (!output1.is_open()){
               std::cout << "could not open file " << filename << std::endl ;
               return false;
    }

    for (auto& x: indexVsContrib) {
    	 output1 << x.first << " " << x.second << '\n';
    }

    output1.close();
    return true;
}



