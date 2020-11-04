#
#  RestrictionEquation.py
#  wMPSHybridCalib
#
#  Created by Zhang Rao on 2020/11/3.
#  Copyright © 2020 Zhang Rao. All rights reserved.
#
from scipy.optimize import least_squares
from wMPSAlgorithm import *
from numpy import append, sqrt


class RestrictionEquation:
    """
    This class is used to introduce the Restriction Equation for
        the first step optimize(ruler length).

        Attributes:
            m_main_transmitter: A Transmitter class for main transmitter .
            m_sub_transmitter: A Transmitter class for sub transmitter.
            m_main_station_time: A (ruler point number , 2) array for saving main transmitter ruler scan time.
            m_sub_station_time: A (ruler point number , 2) array for saving sub transmitter ruler scan time.
            m_ruler_length: A float for saving ruler length.
            m_init_value: A (3+1+point number,3) array for saving calibration init value.
        """

    def __init__(self, main_station, sub_station, ruler_length, init_value):
        self.m_main_transmitter = main_station
        self.m_sub_transmitter = sub_station
        self.m_main_station_time = main_station.m_ruler_calib_scan_time
        self.m_sub_station_time = sub_station.m_ruler_calib_scan_time
        self.m_ruler_length = ruler_length
        self.m_init_value = init_value.T

    def function(self, out_para):
        """
        The Target Function for Least Squares
        :param out_para: The external parameter for transmitter
        :return: Target Function Residual
        """
        point_number = size(self.m_main_station_time, 0)
        out_para = out_para.reshape(-1, 3)
        main_station_point = out_para[4:point_number + 4, :]
        sub_station_point = zeros([point_number, 3])
        result = zeros(12)
        r11 = out_para[0, 0]
        r12 = out_para[0, 1]
        r13 = out_para[0, 2]
        r21 = out_para[1, 0]
        r22 = out_para[1, 1]
        r23 = out_para[1, 2]
        r31 = out_para[2, 0]
        r32 = out_para[2, 1]
        r33 = out_para[2, 2]
        t1 = out_para[3, 0]
        t2 = out_para[3, 1]
        t3 = out_para[3, 2]
        rotation = array([[r11, r12, r13],
                          [r21, r22, r23],
                          [r31, r32, r33]]).T
        transformation = array([t1, t2, t3])
        for i in range(point_number):
            sub_station_point[i, :] = rotation.dot(main_station_point[i, :]) + transformation.T

        # Rotation matrix Normality constraint
        ratio = 1000000
        result[0] = ratio * abs(r11 * r11 + r12 * r12 + r13 * r13 - 1)
        result[1] = ratio * abs(r21 * r21 + r22 * r22 + r23 * r23 - 1)
        result[2] = ratio * abs(r31 * r31 + r32 * r32 + r33 * r33 - 1)
        result[3] = ratio * (r11 * r12 + r21 * r22 + r31 * r32)
        result[4] = ratio * (r11 * r13 + r21 * r23 + r31 * r33)
        result[5] = ratio * (r12 * r13 + r22 * r23 + r32 * r33)
        result[6] = ratio * abs(r11 * r11 + r21 * r21 + r31 * r31 - 1)
        result[7] = ratio * abs(r12 * r12 + r22 * r22 + r32 * r32 - 1)
        result[8] = ratio * abs(r13 * r13 + r23 * r23 + r33 * r33 - 1)
        result[9] = ratio * (r11 * r21 + r12 * r22 + r13 * r23)
        result[10] = ratio * (r11 * r31 + r12 * r32 + r13 * r33)
        result[11] = ratio * (r21 * r31 + r22 * r32 + r23 * r33)

        # Measurement function for wMPS
        for i in range(point_number):
            ct1 = cos(self.m_main_station_time[i, 0] * 2 * pi)
            st1 = sin(self.m_main_station_time[i, 0] * 2 * pi)
            v = array([[ct1, st1, 0], [-st1, ct1, 0], [0, 0, 1]]).dot(self.m_main_transmitter.m_innpara[0:3].T)
            result = append(result, v.T.dot(main_station_point[i, :]) + self.m_main_transmitter.m_innpara[3])
            ct2 = cos(self.m_main_station_time[i, 1] * 2 * pi)
            st2 = sin(self.m_main_station_time[i, 1] * 2 * pi)
            v = array([[ct2, st2, 0], [-st2, ct2, 0], [0, 0, 1]]).dot(self.m_main_transmitter.m_innpara[4:7].T)
            result = append(result, v.T.dot(main_station_point[i, :]) + self.m_main_transmitter.m_innpara[7])
        for i in range(point_number):
            ct1 = cos(self.m_sub_station_time[i, 0] * 2 * pi)
            st1 = sin(self.m_sub_station_time[i, 0] * 2 * pi)
            v = array([[ct1, st1, 0], [-st1, ct1, 0], [0, 0, 1]]).dot(self.m_sub_transmitter.m_innpara[0:3].T)
            result = append(result, v.T.dot(sub_station_point[i, :]) + self.m_sub_transmitter.m_innpara[3])
            ct2 = cos(self.m_sub_station_time[i, 1] * 2 * pi)
            st2 = sin(self.m_sub_station_time[i, 1] * 2 * pi)
            v = array([[ct2, st2, 0], [-st2, ct2, 0], [0, 0, 1]]).dot(self.m_sub_transmitter.m_innpara[4:7].T)
            result = append(result, v.T.dot(sub_station_point[i, :]) + self.m_sub_transmitter.m_innpara[7])

        # The distance Restriction based on ruler length
        for i in range(int(point_number / 2)):
            distance = sum((sub_station_point[2 * i + 1, :] - sub_station_point[2 * i, :]) * (
                    sub_station_point[2 * i + 1, :] - sub_station_point[2 * i, :]))
            result = append(result, sqrt(distance) - self.m_ruler_length)
        return result

    def resolve_function(self):
        """
        Run least_squares for the first step ruler length optimize.
        :return: least_squares results
        """
        return least_squares(self.function, self.m_init_value.reshape(1, -1)[0])


class HybridRestrictionEquation:
    """
    This class is used to introduce the Restriction Equation for
        the second step optimize(several control point + ruler length).

        Attributes:
            :m_main_transmitter: A Transmitter class for main transmitter .
            :m_sub_transmitter: A Transmitter class for sub transmitter.
            :m_main_station_time: A (ruler point number , 2) array for saving main transmitter ruler scan time.
            :m_sub_station_time: A (ruler point number , 2) array for saving sub transmitter ruler scan time.
            :m_main_station_point_time: A (control point number , 2) array for
                saving main transmitter control point scan time.
            :m_sub_station_point_time: A (control point number , 2) array for
                saving sub transmitter control point scan time.
            :m_ruler_length: A float for saving ruler length.
            :m_init_value: A (3+1+point number,3) array for saving calibration init value.
            :m_control_point：A (control point number,3) array for saving control points' coordinate.
        """

    def __init__(self, main_station, sub_station, ruler_length, control_point, init_value):
        self.m_main_transmitter = main_station
        self.m_sub_transmitter = sub_station
        self.m_main_station_time = main_station.m_ruler_calib_scan_time
        self.m_sub_station_time = sub_station.m_ruler_calib_scan_time
        self.m_main_station_point_time = main_station.m_point_calib_scan_time
        self.m_sub_station_point_time = sub_station.m_point_calib_scan_time
        self.m_ruler_length = ruler_length
        self.m_init_value = init_value.T
        self.m_control_point = control_point

    def function(self, out_para):
        """
        The Target Function for Least Squares
        :param out_para: The external parameter for transmitter
        :return: Target Function Residual
        """
        point_ratio = 1000  # The ratio for control point constraints
        ruler_point_number = size(self.m_main_station_time, 0)
        control_point_number = size(self.m_control_point, 0)
        result = zeros(24)

        # Rotation matrix Normality constraint
        result[0:12], main_station_rotation, main_station_transformation \
            = self.rotation_matrix_restrict(out_para.reshape(-1, 3))
        result[12:24], sub_station_rotation, sub_station_transformation \
            = self.rotation_matrix_restrict(out_para[12:].reshape(-1, 3))

        global_ruler_points = out_para[24:].reshape(-1, 3)
        main_station_point = transform_point(global_ruler_points,
                                             main_station_rotation, main_station_transformation)
        sub_station_point = transform_point(global_ruler_points,
                                            sub_station_rotation, sub_station_transformation)
        main_control_point = transform_point(self.m_control_point,
                                             main_station_rotation, main_station_transformation)
        sub_control_point = transform_point(self.m_control_point,
                                            sub_station_rotation, sub_station_transformation)

        # Ruler Point Measurement function for wMPS
        for i in range(ruler_point_number):
            ct1 = cos(self.m_main_station_time[i, 0] * 2 * pi)
            st1 = sin(self.m_main_station_time[i, 0] * 2 * pi)
            v = array([[ct1, st1, 0], [-st1, ct1, 0], [0, 0, 1]]).dot(self.m_main_transmitter.m_innpara[0:3].T)
            result = append(result, v.T.dot(main_station_point[i, :]) + self.m_main_transmitter.m_innpara[3])
            ct2 = cos(self.m_main_station_time[i, 1] * 2 * pi)
            st2 = sin(self.m_main_station_time[i, 1] * 2 * pi)
            v = array([[ct2, st2, 0], [-st2, ct2, 0], [0, 0, 1]]).dot(self.m_main_transmitter.m_innpara[4:7].T)
            result = append(result, v.T.dot(main_station_point[i, :]) + self.m_main_transmitter.m_innpara[7])
        for i in range(ruler_point_number):
            ct1 = cos(self.m_sub_station_time[i, 0] * 2 * pi)
            st1 = sin(self.m_sub_station_time[i, 0] * 2 * pi)
            v = array([[ct1, st1, 0], [-st1, ct1, 0], [0, 0, 1]]).dot(self.m_sub_transmitter.m_innpara[0:3].T)
            result = append(result, v.T.dot(sub_station_point[i, :]) + self.m_sub_transmitter.m_innpara[3])
            ct2 = cos(self.m_sub_station_time[i, 1] * 2 * pi)
            st2 = sin(self.m_sub_station_time[i, 1] * 2 * pi)
            v = array([[ct2, st2, 0], [-st2, ct2, 0], [0, 0, 1]]).dot(self.m_sub_transmitter.m_innpara[4:7].T)
            result = append(result, v.T.dot(sub_station_point[i, :]) + self.m_sub_transmitter.m_innpara[7])

        # The distance Restriction based on ruler length
        for i in range(int(ruler_point_number / 2)):
            distance = sum((sub_station_point[2 * i + 1, :] - sub_station_point[2 * i, :]) * (
                    sub_station_point[2 * i + 1, :] - sub_station_point[2 * i, :]))
            result = append(result, sqrt(distance) - self.m_ruler_length)

        # Control Points Measurement function for wMPS
        for i in range(control_point_number):
            ct1 = cos(self.m_main_station_point_time[i, 0] * 2 * pi)
            st1 = sin(self.m_main_station_point_time[i, 0] * 2 * pi)
            v = array([[ct1, st1, 0], [-st1, ct1, 0], [0, 0, 1]]).dot(self.m_main_transmitter.m_innpara[0:3].T)
            result = append(result, point_ratio * (v.T.dot(main_control_point[i, :])
                                                   + self.m_main_transmitter.m_innpara[3]))
            ct2 = cos(self.m_main_station_point_time[i, 1] * 2 * pi)
            st2 = sin(self.m_main_station_point_time[i, 1] * 2 * pi)
            v = array([[ct2, st2, 0], [-st2, ct2, 0], [0, 0, 1]]).dot(self.m_main_transmitter.m_innpara[4:7].T)
            result = append(result, point_ratio * (v.T.dot(main_control_point[i, :])
                                                   + self.m_main_transmitter.m_innpara[7]))
        for i in range(control_point_number):
            ct1 = cos(self.m_sub_station_point_time[i, 0] * 2 * pi)
            st1 = sin(self.m_sub_station_point_time[i, 0] * 2 * pi)
            v = array([[ct1, st1, 0], [-st1, ct1, 0], [0, 0, 1]]).dot(self.m_sub_transmitter.m_innpara[0:3].T)
            result = append(result, point_ratio * (v.T.dot(sub_control_point[i, :])
                                                   + self.m_sub_transmitter.m_innpara[3]))
            ct2 = cos(self.m_sub_station_point_time[i, 1] * 2 * pi)
            st2 = sin(self.m_sub_station_point_time[i, 1] * 2 * pi)
            v = array([[ct2, st2, 0], [-st2, ct2, 0], [0, 0, 1]]).dot(self.m_sub_transmitter.m_innpara[4:7].T)
            result = append(result, point_ratio * (v.T.dot(sub_control_point[i, :])
                                                   + self.m_sub_transmitter.m_innpara[7]))
        return result

    def resolve_function(self):
        """
        Run least_squares for the Second step hybrid optimize.
        :return: least_squares results
        """
        return least_squares(self.function, self.m_init_value.reshape(1, -1)[0])

    @staticmethod
    def rotation_matrix_restrict(out_para):
        """
        Construct Rotation Matrix Normalize Constraints.
        :param out_para: The external parameter for transmitter
        :return: Normalize Constraints Result, Rotation matrix, Transformation matrix
        """
        result = zeros(12)
        r11 = out_para[0, 0]
        r12 = out_para[0, 1]
        r13 = out_para[0, 2]
        r21 = out_para[1, 0]
        r22 = out_para[1, 1]
        r23 = out_para[1, 2]
        r31 = out_para[2, 0]
        r32 = out_para[2, 1]
        r33 = out_para[2, 2]
        t1 = out_para[3, 0]
        t2 = out_para[3, 1]
        t3 = out_para[3, 2]
        rotation = array([[r11, r12, r13],
                          [r21, r22, r23],
                          [r31, r32, r33]]).T
        transformation = array([t1, t2, t3])
        ratio = 1000000
        result[0] = ratio * abs(r11 * r11 + r12 * r12 + r13 * r13 - 1)
        result[1] = ratio * abs(r21 * r21 + r22 * r22 + r23 * r23 - 1)
        result[2] = ratio * abs(r31 * r31 + r32 * r32 + r33 * r33 - 1)
        result[3] = ratio * (r11 * r12 + r21 * r22 + r31 * r32)
        result[4] = ratio * (r11 * r13 + r21 * r23 + r31 * r33)
        result[5] = ratio * (r12 * r13 + r22 * r23 + r32 * r33)
        result[6] = ratio * abs(r11 * r11 + r21 * r21 + r31 * r31 - 1)
        result[7] = ratio * abs(r12 * r12 + r22 * r22 + r32 * r32 - 1)
        result[8] = ratio * abs(r13 * r13 + r23 * r23 + r33 * r33 - 1)
        result[9] = ratio * (r11 * r21 + r12 * r22 + r13 * r23)
        result[10] = ratio * (r11 * r31 + r12 * r32 + r13 * r33)
        result[11] = ratio * (r21 * r31 + r22 * r32 + r23 * r33)
        return result, rotation, transformation
