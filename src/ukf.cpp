#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true ;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);


  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // TODO: add other variables 
  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = .75; //##### fiddle with this one #####

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = .45; //##### and this one #####

  is_initialized_ = false;
  previous_timestamp_ = 0;

  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;

  x_ = VectorXd(n_x_);
  P_ = MatrixXd(n_x_, n_x_);
  x_aug_ = VectorXd(n_aug_);
  P_aug_ = MatrixXd(n_aug_, n_aug_);
  X_aug_sig_ = MatrixXd(n_aug_, 2*n_aug_+1);
  X_sig_pred_= MatrixXd(n_x_, 2*n_aug_+1);
  Z_aug_sig_ = MatrixXd(3, 2*n_aug_+1);
  weights_ = VectorXd(2*n_aug_+1);


  // ai tog Andre, declare you variables. Ons werk nou met grootmens tale...
  R_laser_ = MatrixXd(2, 2);
  H_laser_ = MatrixXd(2, 5);

  R_laser_ <<   0.0225, 0,
                0,      0.0225;

  H_laser_ <<  1, 0, 0, 0, 0,
              0, 1, 0, 0, 0;


  P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1;

  x_ << 1, 1, 1, 1, 1;




  weights_.fill(0.5/(lambda_ + n_aug_));
  weights_(0) = lambda_ / (lambda_ + n_aug_);

}


// from lesson 18
void UKF::AugmentSigmaPoints()
{
  x_aug_.head(5) = x_;  
  x_aug_(5) = 0;
  x_aug_(6) = 0;

  //create augmented covariance matrix
  P_aug_.fill(0.0);
  P_aug_.topLeftCorner(5,5) = P_;
  P_aug_(5,5) = std_a_*std_a_;
  P_aug_(6,6) = std_yawdd_*std_yawdd_;

  //create square root matrix
  MatrixXd L = P_aug_.llt().matrixL();

  //create augmented sigma points
  X_aug_sig_.col(0)  = x_aug_;
  for (int i = 0; i< n_aug_; i++)
  {
    X_aug_sig_.col(i+1)       = x_aug_ + sqrt(lambda_+n_aug_) * L.col(i);
    X_aug_sig_.col(i+1+n_aug_) = x_aug_ - sqrt(lambda_+n_aug_) * L.col(i);
  }

}

// from lesson 21
void UKF::SigmaPointPrediction()
{

 for (int i = 0; i< 2*n_aug_+1; i++)
  {
    //extract values for better readability
    double p_x = X_aug_sig_(0,i);
    double p_y = X_aug_sig_(1,i);
    double v = X_aug_sig_(2,i);
    double yaw = X_aug_sig_(3,i);
    double yawd = X_aug_sig_(4,i);
    double nu_a = X_aug_sig_(5,i);
    double nu_yawdd = X_aug_sig_(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*dt_) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*dt_) );
    }
    else {
        px_p = p_x + v*dt_*cos(yaw);
        py_p = p_y + v*dt_*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*dt_;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*dt_*dt_ * cos(yaw);
    py_p = py_p + 0.5*nu_a*dt_*dt_ * sin(yaw);
    v_p = v_p + nu_a*dt_;

    yaw_p = yaw_p + 0.5*nu_yawdd*dt_*dt_;
    yawd_p = yawd_p + nu_yawdd*dt_;

    //write predicted sigma point into right column
    X_sig_pred_(0,i) = px_p;
    X_sig_pred_(1,i) = py_p;
    X_sig_pred_(2,i) = v_p;
    X_sig_pred_(3,i) = yaw_p;
    X_sig_pred_(4,i) = yawd_p;

  }

}


// from lesson 24
void UKF::PredictMeanAndCovariance()
{

  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x_ = x_ + weights_(i) * X_sig_pred_.col(i);
  }

  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

    // state difference
    VectorXd x_diff = X_sig_pred_.col(i) - X_sig_pred_.col(0);
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }
}

// from lesson 27
void UKF::PredictRadarMeasurement() 
{
for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    // extract values for better readibility
    double p_x = X_sig_pred_(0,i);
    double p_y = X_sig_pred_(1,i);
    double v  = X_sig_pred_(2,i);
    double yaw = X_sig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Z_aug_sig_(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Z_aug_sig_(1,i) = atan2(p_y,p_x);                                 //phi

    // like the previous project
    if (Z_aug_sig_(0, i) > 0.001) {
      Z_aug_sig_(2, i) = (p_x * v1 + p_y * v2) / Z_aug_sig_(0, i);  
    } else {
      Z_aug_sig_(2, i) = 0.0;  
    }
  }

  //mean predicted measurement
  z_radar_pred_.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
      z_radar_pred_ = z_radar_pred_ + weights_(i) * Z_aug_sig_.col(i);
  }

  //measurement covariance matrix S
  S_radar_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Z_aug_sig_.col(i) - Z_aug_sig_.col(0);

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S_radar_ = S_radar_ + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(3,3);
  R <<    std_radr_*std_radr_, 0, 0,
          0, std_radphi_*std_radphi_, 0,
          0, 0,std_radrd_*std_radrd_;
  S_radar_ = S_radar_ + R;

}

// lesson 30
void UKF::UpdateRadarState()
{

  MatrixXd Tc = MatrixXd(n_x_, 3);

  Tc.fill(0.0);

  for (int i = 1; i < 2*n_aug_+1; i++) {
    VectorXd z_diff = Z_aug_sig_.col(i) - Z_aug_sig_.col(0);

    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    VectorXd x_diff = X_sig_pred_.col(i) - X_sig_pred_.col(0);

    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

   
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  MatrixXd K = Tc * S_radar_.inverse();

  VectorXd z_diff = z_radar_ - z_radar_pred_;


  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  x_ = x_ + K * z_diff;
  P_ = P_ - K*S_radar_*K.transpose();
}


UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */




void UKF::ProcessMeasurement(MeasurementPackage meas_package) {


  if(!is_initialized_)
  {
    previous_timestamp_ = meas_package.timestamp_;
    if(meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
      //init for radar
      float rho = meas_package.raw_measurements_[0];
      float phi = meas_package.raw_measurements_[1];
      float rhodot = meas_package.raw_measurements_[2];

      float x = rho * cos(phi);
      float y = rho * sin(phi);

      x_ << x, y, 0.0, 0.0, 0.0; 
    } else if(meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
      // init for laser
      x_ << meas_package.raw_measurements_[0],
      meas_package.raw_measurements_[1],
      0.0,
      0.0,
      0.0;

    }

  z_laser_pred_ = VectorXd(2);
  z_laser_pred_.fill(0.0);

  z_radar_pred_ = VectorXd(3);
  z_radar_pred_.fill(0.0);

  S_radar_ = MatrixXd(3,3);
  S_radar_.fill(0.0);

  is_initialized_=true;
  }



dt_ = (meas_package.timestamp_ - previous_timestamp_)/1000000.0;
previous_timestamp_ = meas_package.timestamp_;


Prediction(dt_);


if(meas_package.sensor_type_ == MeasurementPackage::RADAR) 
{
  UpdateRadar(meas_package);
} else if(meas_package.sensor_type_ == MeasurementPackage::LASER) 
{
  UpdateLidar(meas_package);
}

}


/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) 
{
  AugmentSigmaPoints();
  SigmaPointPrediction();
  PredictMeanAndCovariance();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) 
{
  z_laser_ = meas_package.raw_measurements_;
  z_laser_pred_ = H_laser_ * x_;
  VectorXd y = z_laser_ - z_laser_pred_;
  MatrixXd Ht = H_laser_.transpose();
  MatrixXd S = H_laser_ * P_ * Ht + R_laser_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  MatrixXd I = MatrixXd::Identity(n_x_, n_x_);
  P_ = (I - K * H_laser_) * P_;


  // update NUS
  NIS_laser_ = (z_laser_ - z_laser_pred_).transpose()
             *S.inverse()
             *(z_laser_ - z_laser_pred_);
               cout << "LASER NIS| " << NIS_laser_ << endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) 
{
  z_radar_ = meas_package.raw_measurements_;
  PredictRadarMeasurement();
  UpdateRadarState();
  
  // update NIS
  NIS_radar_ = (z_radar_ - z_radar_pred_).transpose()
               *S_radar_.inverse()
               *(z_radar_ - z_radar_pred_);

  cout << "RADAR NIS| " << NIS_radar_ << endl;
}
