import numpy as np

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


def main():
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[0.],
                     [0.]])  # initial state (location and velocity)

    kf.F = np.array([[1., 0.],
                     [0., 1.]])  # state transition matrix
    kf.H = np.array([[1., 0.]])  # Measurement function
    kf.P *= 1000.  # covariance matrix
    kf.R = np.array([[1e-10]])  # measurement noise
    # kf.Q = Q_discrete_white_noise(2, 1/30, .1) # process uncertainty
    kf.Q *= 1e10
    print(kf.Q)

    kf.update(np.array([1.]))
    kf.predict()
    print(kf.x)

    kf.update(np.array([2.]))
    kf.predict()
    print(kf.x)

    kf.update(np.array([1.]))
    kf.predict()
    print(kf.x)

    kf.update(np.array([0.]))
    kf.predict()
    print(kf.x)

    kf.update(np.array([5.]))
    kf.predict()
    print(kf.x)


if __name__ == '__main__':
    main()
