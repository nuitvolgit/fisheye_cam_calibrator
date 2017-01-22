#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdio>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <ros/ros.h>

class FisheyeCalib {
public:
  // private ROS node handle
  ros::NodeHandle node_;
  FisheyeCalib() : goodInput(false), calibrationPattern(NOT_EXISTING) {}
  ~FisheyeCalib() {}

  enum Pattern {NOT_EXISTING, CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID};
  enum InputType {INVALID, CAMERA, VIDEO_FILE, IMAGE_LIST};

  void read(const cv::FileNode& node)                          //Read serialization for this class
  {    
      node["BoardSize_Width" ] >> boardSize.width;
      node["BoardSize_Height"] >> boardSize.height;
      node["Calibrate_Pattern"] >> patternToUse;
      node["Square_Size"]  >> squareSize;
      node["Calibrate_NrOfFrameToUse"] >> nrFrames;
      node["Calibrate_FixAspectRatio"] >> aspectRatio;
      node["Write_DetectedFeaturePoints"] >> writePoints;
      node["Write_extrinsicParameters"] >> writeExtrinsics;
      node["Write_outputFileName"] >> outputFileName;
      node["Calibrate_AssumeZeroTangentialDistortion"] >> calibZeroTangentDist;
      node["Calibrate_FixPrincipalPointAtTheCenter"] >> calibFixPrincipalPoint;
      node["Calibrate_UseFisheyeModel"] >> useFisheye;
      node["Input_FlipAroundHorizontalAxis"] >> flipVertical;
      node["Show_UndistortedImage"] >> showUndistorsed;
      node["Input"] >> input;
      node["Input_Delay"] >> delay;
      node["Fix_K1"] >> fixK1;
      node["Fix_K2"] >> fixK2;
      node["Fix_K3"] >> fixK3;
      node["Fix_K4"] >> fixK4;
      node["Fix_K5"] >> fixK5;
      validate();
  }

  void validate() {
    goodInput = true;
    if (boardSize.width <= 0 || boardSize.height <= 0) {
      std::cerr << "Invalid Board size: " << boardSize.width << " " << boardSize.height << std::endl;
      goodInput = false;
    }
    if (squareSize <= 10e-6) {
      std::cerr << "Invalid square size " << squareSize << std::endl;
      goodInput = false;
    }
    if (nrFrames <= 0) {
      std::cerr << "Invalid number of frames " << nrFrames << std::endl;
      goodInput = false;
    }
    if (input.empty()) {      // Check for valid input
      inputType = INVALID;
    } else {
      if (input[0] >= '0' && input[0] <= '9') { // Image from cameras
        std::stringstream ss(input);
        ss >> cameraID;
        inputType = CAMERA;
      } else {
        if (readStringList(input, imageList)) { // Image from files
          inputType = IMAGE_LIST;
          nrFrames = (nrFrames < (int)imageList.size()) ? nrFrames : (int)imageList.size();
        } else {                                // Image from video
          inputType = VIDEO_FILE;
        }
      }

      if (inputType == CAMERA)
        inputCapture.open(cameraID);
      else if (inputType == VIDEO_FILE)
        inputCapture.open(input);
      else if (inputType != IMAGE_LIST && !inputCapture.isOpened())
        inputType = INVALID;
    }

    if (inputType == INVALID) {
      std::cerr << " Input does not exist: " << input;
      goodInput = false;
    }

    flag = 0;
    if (calibFixPrincipalPoint) flag |= cv::CALIB_FIX_PRINCIPAL_POINT;
    if (calibZeroTangentDist)   flag |= cv::CALIB_ZERO_TANGENT_DIST;
    if (aspectRatio)            flag |= cv::CALIB_FIX_ASPECT_RATIO;
    if (fixK1)                  flag |= cv::CALIB_FIX_K1;
    if (fixK2)                  flag |= cv::CALIB_FIX_K2;
    if (fixK3)                  flag |= cv::CALIB_FIX_K3;
    if (fixK4)                  flag |= cv::CALIB_FIX_K4;
    if (fixK5)                  flag |= cv::CALIB_FIX_K5;

    if (useFisheye) {
      // the fisheye model has its own enum, so overwrite the flags
      flag = cv::fisheye::CALIB_FIX_SKEW | cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
      if (fixK1)                flag |= cv::fisheye::CALIB_FIX_K1;
      if (fixK2)                flag |= cv::fisheye::CALIB_FIX_K2;
      if (fixK3)                flag |= cv::fisheye::CALIB_FIX_K3;
      if (fixK4)                flag |= cv::fisheye::CALIB_FIX_K4;
    }

    if (!patternToUse.compare("CHESSBOARD"))
      calibrationPattern = CHESSBOARD;
    else if (!patternToUse.compare("CIRCLES_GRID"))
      calibrationPattern = CIRCLES_GRID;
    else if (!patternToUse.compare("ASYMMETRIC_CIRCLES_GRID"))
      calibrationPattern = ASYMMETRIC_CIRCLES_GRID;
    else {
      std::cerr << " Camera calibration mode does not exist: " << patternToUse << std::endl;
      goodInput = false;
    }
    atImageList = 0;
  }

  /*
  static bool readStringList(const std::string& filename, std::vector<std::string>& l ) {
    l.clear();
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened())
      return false;
    cv::FileNode n = fs.getFirstTopLevelNode();
    if(n.type() != cv::FileNode::SEQ )
      return false;
    cv::FileNodeIterator it = n.begin(), it_end = n.end();
    for( ; it != it_end; ++it )
      l.push_back((std::string)*it);
    return true;
  }
  */
  static bool readStringList(const std::string& filename, std::vector<std::string>& l ) {
    l.clear();
    cv::FileStorage fs(filename, cv::FileStorage::READ);

    if (!fs.isOpened())
      return false;

    std::string imagePath, imageName, imageType;
    int imageNum;
    fs["Image_path"] >> imagePath;
    fs["Image_name"] >> imageName;
    fs["Image_type"] >> imageType;
    fs["Image_num"] >> imageNum;
    fs.release();

    if (imageNum < 1)
      return false;

    for (int i = 1; i <= imageNum; ++i)
      l.push_back(imagePath + imageName + std::to_string(i) + "." + imageType);

    return true;
  }

  void write(cv::FileStorage& fs) const { //Write serialization for this class
    fs << "{"
              << "BoardSize_Width"  << boardSize.width
              << "BoardSize_Height" << boardSize.height
              << "Square_Size"         << squareSize
              << "Calibrate_Pattern" << patternToUse
              << "Calibrate_NrOfFrameToUse" << nrFrames
              << "Calibrate_FixAspectRatio" << aspectRatio
              << "Calibrate_AssumeZeroTangentialDistortion" << calibZeroTangentDist
              << "Calibrate_FixPrincipalPointAtTheCenter" << calibFixPrincipalPoint

              << "Write_DetectedFeaturePoints" << writePoints
              << "Write_extrinsicParameters"   << writeExtrinsics
              << "Write_outputFileName"  << outputFileName

              << "Show_UndistortedImage" << showUndistorsed

              << "Input_FlipAroundHorizontalAxis" << flipVertical
              << "Input_Delay" << delay
              << "Input" << input
    << "}";
  }

  cv::Mat nextImage() {
    cv::Mat result;
    if (inputCapture.isOpened()) {
      cv::Mat view0;
      inputCapture >> view0;
      view0.copyTo(result);
    } else if(atImageList < imageList.size()) {
      result = cv::imread(imageList[atImageList++], cv::IMREAD_COLOR);
    }
    return result;
  }


public:
  cv::Size boardSize;              // The size of the board -> Number of items by width and height
  Pattern calibrationPattern;  // One of the Chessboard, circles, or asymmetric circle pattern
  float squareSize;            // The size of a square in your defined unit (point, millimeter,etc).
  int nrFrames;                // The number of frames to use from the input for calibration
  float aspectRatio;           // The aspect ratio
  int delay;                   // In case of a video input
  bool writePoints;            // Write detected feature points
  bool writeExtrinsics;        // Write extrinsic parameters
  bool calibZeroTangentDist;   // Assume zero tangential distortion
  bool calibFixPrincipalPoint; // Fix the principal point at the center
  bool flipVertical;           // Flip the captured images around the horizontal axis
  bool showUndistorsed;        // Show undistorted images after calibration
  std::string outputFileName;       // The name of the file where to write
  std::string input;                // The input ->
  bool useFisheye;             // use fisheye camera model for calibration
  bool fixK1;                  // fix K1 distortion coefficient
  bool fixK2;                  // fix K2 distortion coefficient
  bool fixK3;                  // fix K3 distortion coefficient
  bool fixK4;                  // fix K4 distortion coefficient
  bool fixK5;                  // fix K5 distortion coefficient

  int cameraID;
  std::vector<std::string> imageList;
  std::size_t atImageList;
  cv::VideoCapture inputCapture;
  InputType inputType;
  bool goodInput;
  int flag;

private:
  std::string patternToUse;

};



enum { DETECTION = 0, CAPTURING = 1, CALIBRATED = 2 };

static inline void write(cv::FileStorage& fs, const cv::String&, const FisheyeCalib& fish_cal) {
  //fish_cal.write(fs);
  std::cout << "Come here??\n";
}

static inline void read(const cv::FileNode& node, FisheyeCalib& fish_cal,
                        const FisheyeCalib& default_value = FisheyeCalib()) {  
  if (node.empty())
    fish_cal = default_value;
  else
    fish_cal.read(node);
}

static void calcBoardCornerPositions(cv::Size boardSize, float squareSize, std::vector<cv::Point3f>& corners,
                                     FisheyeCalib::Pattern patternType /*= Settings::CHESSBOARD*/) {
  corners.clear();
  switch(patternType) {
    case FisheyeCalib::CHESSBOARD:
    case FisheyeCalib::CIRCLES_GRID:
      for (int i = 0; i < boardSize.height; ++i) {
        for (int j = 0; j < boardSize.width; ++j) {
          corners.push_back(cv::Point3f(j*squareSize, i*squareSize, 0));
        }
      }
      break;
    case FisheyeCalib::ASYMMETRIC_CIRCLES_GRID:
      for (int i = 0; i < boardSize.height; i++) {
        for(int j = 0; j < boardSize.width; j++) {
          corners.push_back(cv::Point3f((2*j + i % 2)*squareSize, i*squareSize, 0));
        }
      }
      break;
    default:
      break;
  }
}


//! [compute_errors]
static double computeReprojectionErrors(const std::vector< std::vector<cv::Point3f> >& objectPoints,
                                        const std::vector< std::vector<cv::Point2f> >& imagePoints,
                                        const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs,
                                        const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs,
                                        std::vector<float>& perViewErrors, bool fisheye) {
  std::vector<cv::Point2f> imagePoints2;
  std::size_t totalPoints = 0;
  double totalErr = 0, err;
  perViewErrors.resize(objectPoints.size());

  for (std::size_t i = 0; i < objectPoints.size(); ++i ) {
    if (fisheye) {
      cv::fisheye::projectPoints(objectPoints[i], imagePoints2, rvecs[i], tvecs[i], cameraMatrix, distCoeffs);
    } else {
      cv::projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePoints2);
    }
    err = norm(imagePoints[i], imagePoints2, cv::NORM_L2);

    std::size_t n = objectPoints[i].size();
    perViewErrors[i] = (float) std::sqrt(err*err/n);
    totalErr        += err*err;
    totalPoints     += n;
  }
  return std::sqrt(totalErr/totalPoints);
}

// Print camera parameters to the output file
static void saveCameraParams(FisheyeCalib& s, cv::Size& imageSize, cv::Mat& cameraMatrix, cv::Mat& distCoeffs,
                             const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs,
                             const std::vector<float>& reprojErrs,
                             const std::vector< std::vector<cv::Point2f> >& imagePoints,
                             double totalAvgErr) {
  cv::FileStorage fs(s.outputFileName, cv::FileStorage::WRITE);
  std::time_t tm;
  std::time(&tm);
  struct tm *t2 = std::localtime(&tm);
  char buf[1024];
  strftime(buf, sizeof(buf), "%c", t2);

  fs << "calibration_time" << buf;

  if( !rvecs.empty() || !reprojErrs.empty() )
      fs << "nr_of_frames" << (int)std::max(rvecs.size(), reprojErrs.size());
  fs << "image_width" << imageSize.width;
  fs << "image_height" << imageSize.height;
  fs << "board_width" << s.boardSize.width;
  fs << "board_height" << s.boardSize.height;
  fs << "square_size" << s.squareSize;

  if(s.flag & cv::CALIB_FIX_ASPECT_RATIO )
    fs << "fix_aspect_ratio" << s.aspectRatio;

  if (s.flag) {
    std::stringstream flagsStringStream;
    if (s.useFisheye) {
      flagsStringStream << "flags:"
          << (s.flag & cv::fisheye::CALIB_FIX_SKEW ? " +fix_skew" : "")
          << (s.flag & cv::fisheye::CALIB_FIX_K1 ? " +fix_k1" : "")
          << (s.flag & cv::fisheye::CALIB_FIX_K2 ? " +fix_k2" : "")
          << (s.flag & cv::fisheye::CALIB_FIX_K3 ? " +fix_k3" : "")
          << (s.flag & cv::fisheye::CALIB_FIX_K4 ? " +fix_k4" : "")
          << (s.flag & cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC ? " +recompute_extrinsic" : "");
    } else {
      flagsStringStream << "flags:"
          << (s.flag & cv::CALIB_USE_INTRINSIC_GUESS ? " +use_intrinsic_guess" : "")
          << (s.flag & cv::CALIB_FIX_ASPECT_RATIO ? " +fix_aspectRatio" : "")
          << (s.flag & cv::CALIB_FIX_PRINCIPAL_POINT ? " +fix_principal_point" : "")
          << (s.flag & cv::CALIB_ZERO_TANGENT_DIST ? " +zero_tangent_dist" : "")
          << (s.flag & cv::CALIB_FIX_K1 ? " +fix_k1" : "")
          << (s.flag & cv::CALIB_FIX_K2 ? " +fix_k2" : "")
          << (s.flag & cv::CALIB_FIX_K3 ? " +fix_k3" : "")
          << (s.flag & cv::CALIB_FIX_K4 ? " +fix_k4" : "")
          << (s.flag & cv::CALIB_FIX_K5 ? " +fix_k5" : "");
    }
    fs.writeComment(flagsStringStream.str());
  }

  fs << "flags" << s.flag;

  fs << "fisheye_model" << s.useFisheye;

  fs << "camera_matrix" << cameraMatrix;
  fs << "distortion_coefficients" << distCoeffs;

  fs << "avg_reprojection_error" << totalAvgErr;
  if (s.writeExtrinsics && !reprojErrs.empty())
      fs << "per_view_reprojection_errors" << cv::Mat(reprojErrs);

  if (s.writeExtrinsics && !rvecs.empty() && !tvecs.empty()) {
    CV_Assert(rvecs[0].type() == tvecs[0].type());
    cv::Mat bigmat((int)rvecs.size(), 6, CV_MAKETYPE(rvecs[0].type(), 1));
    bool needReshapeR = rvecs[0].depth() != 1 ? true : false;
    bool needReshapeT = tvecs[0].depth() != 1 ? true : false;

    for(std::size_t i = 0; i < rvecs.size(); i++) {
      cv::Mat r = bigmat(cv::Range(int(i), int(i+1)), cv::Range(0,3));
      cv::Mat t = bigmat(cv::Range(int(i), int(i+1)), cv::Range(3,6));
      if (needReshapeR) {
        rvecs[i].reshape(1, 1).copyTo(r);
      } else {
        //*.t() is MatExpr (not Mat) so we can use assignment operator
        CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
        r = rvecs[i].t();
      }

      if(needReshapeT) {
        tvecs[i].reshape(1, 1).copyTo(t);
      } else {
        CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
        t = tvecs[i].t();
      }
    }
    fs.writeComment("a set of 6-tuples (rotation vector + translation vector) for each view");
    fs << "extrinsic_parameters" << bigmat;
  }

  if (s.writePoints && !imagePoints.empty()) {
    cv::Mat imagePtMat((int)imagePoints.size(), (int)imagePoints[0].size(), CV_32FC2);
    for (std::size_t i = 0; i < imagePoints.size(); i++) {
      cv::Mat r = imagePtMat.row(int(i)).reshape(2, imagePtMat.cols);
      cv::Mat imgpti(imagePoints[i]);
      imgpti.copyTo(r);
    }
    fs << "image_points" << imagePtMat;
  }
}


//! [board_corners]
static bool runCalibration(FisheyeCalib& fc, cv::Size& imageSize, cv::Mat& cameraMatrix, cv::Mat& distCoeffs,
                           std::vector< std::vector<cv::Point2f> > imagePoints, std::vector<cv::Mat>& rvecs,
                           std::vector<cv::Mat>& tvecs, std::vector<float>& reprojErrs,  double& totalAvgErr) {
  //! [fixed_aspect]
  cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
  if (fc.flag & cv::CALIB_FIX_ASPECT_RATIO )
    cameraMatrix.at<double>(0,0) = fc.aspectRatio;

  //! [fixed_aspect]
  if (fc.useFisheye) {
    distCoeffs = cv::Mat::zeros(4, 1, CV_64F);
  } else {
    distCoeffs = cv::Mat::zeros(8, 1, CV_64F);
  }

  std::vector< std::vector<cv::Point3f> > objectPoints(1);
  calcBoardCornerPositions(fc.boardSize, fc.squareSize, objectPoints[0], fc.calibrationPattern);

  objectPoints.resize(imagePoints.size(),objectPoints[0]);

  //Find intrinsic and extrinsic camera parameters
  double rms;

  if (fc.useFisheye) {
    cv::Mat _rvecs, _tvecs;
    rms = cv::fisheye::calibrate(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs,
                                 _rvecs, _tvecs, fc.flag);
    rvecs.reserve(_rvecs.rows);
    tvecs.reserve(_tvecs.rows);
    for(int i = 0; i < int(objectPoints.size()); i++){
        rvecs.push_back(_rvecs.row(i));
        tvecs.push_back(_tvecs.row(i));
    }
  } else {
    rms = cv::calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs,
                          rvecs, tvecs, fc.flag);
  }

  std::cout << "Re-projection error reported by calibrateCamera: "<< rms << std::endl;

  bool ok = cv::checkRange(cameraMatrix) && checkRange(distCoeffs);

  totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints, rvecs, tvecs, cameraMatrix,
                                          distCoeffs, reprojErrs, fc.useFisheye);

  return ok;
}

bool runCalibrationAndSave(FisheyeCalib& fc, cv::Size imageSize, cv::Mat& cameraMatrix, cv::Mat& distCoeffs,
                           std::vector< std::vector<cv::Point2f> > imagePoints) {
  std::vector<cv::Mat> rvecs, tvecs;
  std::vector<float> reprojErrs;
  double totalAvgErr = 0;

  bool ok = runCalibration(fc, imageSize, cameraMatrix, distCoeffs, imagePoints, rvecs, tvecs,
                           reprojErrs, totalAvgErr);
  std::cout << (ok ? "Calibration succeeded" : "Calibration failed")
            << ". avg re projection error = " << totalAvgErr << std::endl;


  if (ok)
    saveCameraParams(fc, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, reprojErrs, imagePoints, totalAvgErr);

  return ok;
}




int main(int argc, char **argv) {
  ros::init(argc, argv, "fisheye_calib");

  /*
  // Reading camera matrix and undistort imag
  // Read camera parameters
  const std::string inputCalibFile("/home/dh/qt_catkin_ws/src/fisheye_calib/out/out_camera_data.xml");
  cv::FileStorage fs(inputCalibFile, cv::FileStorage::READ); // Read the settings
  if (!fs.isOpened()) {
    std::cout << "Could not open the camera parameter file: \"" << inputCalibFile << "\"" << std::endl;
    return false;
  }
  cv::Mat camMat, distCoeffs;
  fs["camera_matrix"] >> camMat;
  fs["distortion_coefficients"] >> distCoeffs;
  fs.release();

  // Undistort image
  cv::Mat view = cv::imread("/home/dh/qt_catkin_ws/src/fisheye_calib/img/chk1.jpg", cv::IMREAD_COLOR);
  cv::Size imageSize = view.size();  // Format input image.

  cv::Mat newCamMat;
  cv::fisheye::estimateNewCameraMatrixForUndistortRectify(camMat, distCoeffs, imageSize,
                                                          cv::Matx33d::eye(), newCamMat, 1);

  // for fast processing, such that if you want to undistort video in real-time,
  // it is better to divide 'undistortImage' into two functions:
  // 'initUndistortRectifyMap' and 'remap'
  // and only run 'remap' funciton for each image since map1 and map2 are the same for every image
  // (just like the camara matrix of the undistorted image can be applied to any distorted images)
  cv::Mat temp = view.clone();
  cv::fisheye::undistortImage(temp, view, camMat, distCoeffs, newCamMat);

  cv::imshow("Image View", view);
  cv::waitKey(0);
  */


  FisheyeCalib fcal;
  const std::string inputSettingsFile("/home/dh/qt_catkin_ws/src/fisheye_calib/param/in_VID5.xml");
  cv::FileStorage fs(inputSettingsFile, cv::FileStorage::READ); // Read the settings
  if (!fs.isOpened()) {
    std::cout << "Could not open the configuration file: \"" << inputSettingsFile << "\"" << std::endl;
    return false;
  }

  fs["Settings"] >> fcal; // readl all parameters from the input setting file
  fs.release();           // after reading parameters from the file, close it by destructing.

  if (!fcal.goodInput) {
      std::cout << "Invalid input detected. Application stopping. " << std::endl;
      return false;
  }

  std::vector< std::vector<cv::Point2f> > imagePoints;  // Set of corner pixels of each image
  cv::Mat cameraMatrix, distCoeffs;                     // Final output
  cv::Size imageSize;

  int mode = fcal.inputType == FisheyeCalib::IMAGE_LIST ? CAPTURING : DETECTION;
  clock_t prevTimestamp = 0;
  const cv::Scalar RED(0,0,255), GREEN(0,255,0);
  const char ESC_KEY = 27;

  for (;;) {
    cv::Mat view;
    bool blinkOutput = false;
    view = fcal.nextImage();

    //-----  If no more image, or got enough, then stop calibration and show result -------------
    if (mode == CAPTURING && imagePoints.size() >= (size_t)fcal.nrFrames) {
      if (runCalibrationAndSave(fcal, imageSize, cameraMatrix, distCoeffs, imagePoints))
        mode = CALIBRATED;
      else
        mode = DETECTION;
    }

    // In the case of checking all image lists, or camera does not provide image, this happens
    if (view.empty()) {          // If there are no more images stop the loop
      // if calibration threshold was not reached yet, calibrate now
      if(mode != CALIBRATED && !imagePoints.empty())
        runCalibrationAndSave(fcal, imageSize, cameraMatrix, distCoeffs, imagePoints);
      break;
    }

    imageSize = view.size();  // Format input image.
    if (fcal.flipVertical)   cv::flip(view, view, 0);

    //! [find_pattern]
    std::vector<cv::Point2f> pointBuf;
    bool found;
    int chessBoardFlags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;

    if(!fcal.useFisheye) {
      // fast check erroneously fails with high distortions like fisheye
      chessBoardFlags |= cv::CALIB_CB_FAST_CHECK;
    }

    switch (fcal.calibrationPattern) { // Find feature points on the input format
    case FisheyeCalib::CHESSBOARD:
      found = cv::findChessboardCorners(view, fcal.boardSize, pointBuf, chessBoardFlags);
      break;
    case FisheyeCalib::CIRCLES_GRID:
      found = cv::findCirclesGrid(view, fcal.boardSize, pointBuf);
      break;
    case FisheyeCalib::ASYMMETRIC_CIRCLES_GRID:
      found = cv::findCirclesGrid(view, fcal.boardSize, pointBuf, cv::CALIB_CB_ASYMMETRIC_GRID);
      break;
    default:
      found = false;
      break;
    }

    if (found) {                // If done with success,
      // improve the found corners' coordinate accuracy for chessboard
      if (fcal.calibrationPattern == FisheyeCalib::CHESSBOARD) {
        cv::Mat viewGray;
        cv::cvtColor(view, viewGray, cv::COLOR_BGR2GRAY);
        cv::cornerSubPix(viewGray, pointBuf, cv::Size(11,11), cv::Size(-1,-1),
                     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1 ));
      }

      if (mode == CAPTURING &&  // For camera only take new samples after delay time
          (!fcal.inputCapture.isOpened() || std::clock() - prevTimestamp > fcal.delay * 1e-3 * CLOCKS_PER_SEC) ) {
        imagePoints.push_back(pointBuf);
        prevTimestamp = clock();
        blinkOutput = fcal.inputCapture.isOpened();
      }
      // Draw the corners.
      cv::drawChessboardCorners(view, fcal.boardSize, cv::Mat(pointBuf), found);
    }

    //! [pattern_found]
    //----------------------------- Output Text ------------------------------------------------
    //! [output_text]
    std::string msg = (mode == CAPTURING) ? "100/100" : mode == CALIBRATED ? "Calibrated" : "Press 'g' to start";
    int baseLine = 0;
    cv::Size textSize = cv::getTextSize(msg, 1, 1, 1, &baseLine);
    cv::Point textOrigin(view.cols - 2*textSize.width - 10, view.rows - 2*baseLine - 10);

    if (mode == CAPTURING) {
      if(fcal.showUndistorsed)
        msg = cv::format("%d/%d Undist", (int)imagePoints.size(), fcal.nrFrames );
      else
        msg = cv::format("%d/%d", (int)imagePoints.size(), fcal.nrFrames );
    }

    cv::putText(view, msg, textOrigin, 1, 1, mode == CALIBRATED ?  GREEN : RED);

    if (blinkOutput)
      bitwise_not(view, view);
    //! [output_text]
    //------------------------- Video capture  output  undistorted ------------------------------
    //! [output_undistorted]
    if (mode == CALIBRATED && fcal.showUndistorsed) {
      cv::Mat temp = view.clone();
      if (fcal.useFisheye)
        cv::fisheye::undistortImage(temp, view, cameraMatrix, distCoeffs);
      else
        cv::undistort(temp, view, cameraMatrix, distCoeffs);
    }
    //! [output_undistorted]
    //------------------------------ Show image and check for input commands -------------------
    //! [await_input]
    cv::imshow("Image View", view);
    //cv::waitKey(0); // In the case of using image lists, use this to check whether
                      // the features on the chessboard have detected
    char key = (char)cv::waitKey(fcal.inputCapture.isOpened() ? 50 : fcal.delay);

    if (key  == ESC_KEY)
      break;

    if (key == 'u' && mode == CALIBRATED )
      fcal.showUndistorsed = !fcal.showUndistorsed;

    if (fcal.inputCapture.isOpened() && key == 'g') {
      mode = CAPTURING;
      imagePoints.clear();
    }
    //! [await_input]

  }




  // -----------------------Show the undistorted image for the image list ------------------------
  //! [show_results]
  if (fcal.inputType == FisheyeCalib::IMAGE_LIST && fcal.showUndistorsed) {
    cv::Mat view, rview, map1, map2;
    if (fcal.useFisheye) {
      cv::Mat newCamMat;
      cv::fisheye::estimateNewCameraMatrixForUndistortRectify(cameraMatrix, distCoeffs, imageSize,
                                                              cv::Matx33d::eye(), newCamMat, 1);
      cv::fisheye::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Matx33d::eye(), newCamMat, imageSize,
                                           CV_16SC2, map1, map2);
    } else {
      cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(),
                                  getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
                                  imageSize, CV_16SC2, map1, map2);
    }

    for(std::size_t i = 0; i < fcal.imageList.size(); i++) {
      view = cv::imread(fcal.imageList[i], cv::IMREAD_COLOR);
      if(view.empty())
        continue;
      remap(view, rview, map1, map2, cv::INTER_LINEAR);
      cv::imshow("Image View", rview);
      char c = (char)cv::waitKey();
      if(c  == ESC_KEY || c == 'q' || c == 'Q' )
        break;
    }
  }
  //! [show_results]

  return 0;
}






























