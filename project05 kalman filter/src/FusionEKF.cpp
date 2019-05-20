#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;


FusionEKF::FusionEKF() {
    is_initialized_ = false;
    previous_timestamp_ = 0;

    R_laser_ = MatrixXd(2, 2);
    R_laser_ <<
        0.0225, 0,
        0, 0.0225;

    R_radar_ = MatrixXd(3, 3);
    R_radar_ <<
        0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

    H_laser_ = MatrixXd(2, 4);
    H_laser_ <<
        1, 0, 0, 0,
        0, 1, 0, 0;

    Hj_ = MatrixXd(3, 4);
    Hj_ <<
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0;

    ekf_.P_ = MatrixXd(4, 4);
    ekf_.Q_ = MatrixXd(4, 4);
    ekf_.F_ = MatrixXd(4, 4);
    ekf_.x_ = VectorXd(4);
    ekf_.P_ <<
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1000, 0,
        0, 0, 0, 1000;
    ekf_.F_ <<
        1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1;
    ekf_.Q_ <<
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0;
    ekf_.x_ << 1, 1, 1, 1;
}


/**
* Destructor.
*/
FusionEKF::~FusionEKF() = default;

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
    // ---------------------------------
    // Init
    // ---------------------------------
    if (!is_initialized_) {
        if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
            double rho= measurement_pack.raw_measurements_(0);
            double rho_dot = measurement_pack.raw_measurements_(2);
            double theta = measurement_pack.raw_measurements_(1);

            double px = rho * cos(theta);
            double py = rho * sin(theta);
            double vx = rho_dot * cos(theta);
            double vy = rho_dot * sin(theta);

            ekf_.x_ << px, py, vx, vy;
        } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
            double px = measurement_pack.raw_measurements_(0);
            double py = measurement_pack.raw_measurements_(1);

            ekf_.x_ << px, py, 0.0, 0.0;
        }

        previous_timestamp_ = measurement_pack.timestamp_;
        is_initialized_ = true;
        return;
    }

    // ---------------------------------
    // Predict
    // ---------------------------------
    double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
    double dt_sq = dt * dt;  // dt^2
    double dt_cube = dt * dt * dt;  // dt^3
    double dt_quad = dt * dt * dt * dt;  // dt^4

    // Transition matrix
    ekf_.F_ = MatrixXd(4, 4);
    ekf_.F_ <<
        1, 0, dt, 0,
        0, 1, 0, dt,
        0, 0, 1, 0,
        0, 0, 0, 1;

    // Covariance matrix
    double noise_ax = 9.0;
    double noise_ay = 9.0;
    ekf_.Q_ = MatrixXd(4, 4);
    ekf_.Q_ <<
        dt_quad / 4 * noise_ax, 0,                      dt_cube / 2 * noise_ax, 0,
        0,                      dt_quad / 4 * noise_ay, 0,                      dt_cube / 2 * noise_ay,
        dt_cube / 2 * noise_ax, 0,                      dt_sq * noise_ax,       0,
        0,                      dt_cube / 2 * noise_ay, 0,                      dt_sq * noise_ay;

    previous_timestamp_ = measurement_pack.timestamp_;
    ekf_.Predict();

    // ---------------------------------
    // Update
    // ---------------------------------
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
        ekf_.R_ = R_radar_;
        ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
        ekf_.UpdateEKF(measurement_pack.raw_measurements_);
    } else {
        ekf_.R_ = R_laser_;
        ekf_.H_ = H_laser_;
        ekf_.Update(measurement_pack.raw_measurements_);
    }

    // print the output
    cout << "x_ = " << ekf_.x_ << endl;
    cout << "P_ = " << ekf_.P_ << endl;
}
