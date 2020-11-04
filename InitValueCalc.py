#
#  InitValueCalc.py
#  wMPSHybridCalib
#
#  Created by Zhang Rao on 2020/11/3.
#  Copyright © 2020 Zhang Rao. All rights reserved.
#
from numpy import size, zeros, array, cos, sin, pi, tan, cross, sqrt, arcsin, arctan, \
    append, abs, float64


class InitValueCalc:
    """
    This class is used to calculate the init value for calibration.

        Attributes:
            m_ruler_length: A float for saving ruler length.
            m_raw_scan_time: A (2 * point number, 2) array for saving mixed scan time.
            m_point_number: A int for saving ruler point number.
            m_main_transmitter: A Transmitter class for main transmitter.
            m_sub_transmitter: A Transmitter class for sub transmitter.
            m_main_station_point: A (point number ,3) array for saving ruler point under main station coordinate system.
            m_sub_station_point: A (point number ,3) array for saving ruler point under sub station coordinate system.
        """

    def __init__(self, ruler_length, raw_scan_time, main_transmitter, sub_transmitter):
        self.m_ruler_length = ruler_length
        self.m_raw_scan_time = raw_scan_time
        self.m_point_number = int(size(self.m_raw_scan_time) / 2)
        self.m_main_transmitter = main_transmitter
        self.m_sub_transmitter = sub_transmitter
        self.m_main_station_point = array([])
        self.m_sub_station_point = array([])
        self.handle_raw_scan_time()
        self.cal_main_station_init_value()
        self.cal_sub_station_init_value()

    def handle_raw_scan_time(self):
        """
        Saving raw scan time to different transmitter class.
        :return:
        """
        for i in range(self.m_point_number * 2):
            if i % 2:
                self.m_sub_transmitter.ruler_calib_scan_time_append(array(self.m_raw_scan_time[i].split(),
                                                                          float64))
            else:
                self.m_main_transmitter.ruler_calib_scan_time_append(array(self.m_raw_scan_time[i].split(),
                                                                           float64))

    def cal_main_station_init_value(self):
        """
        Calculate the init value for main_station(ruler point coordinate)
        :return:
        """
        hor_angle = zeros(self.m_point_number)
        ver_angle = zeros(self.m_point_number)
        for i in range(self.m_point_number):
            first_scan_time = self.m_main_transmitter.m_ruler_calib_scan_time[i, 0] * 2 * pi
            second_scan_time = self.m_main_transmitter.m_ruler_calib_scan_time[i, 1] * 2 * pi
            first_plane_vector = array([[cos(first_scan_time), sin(first_scan_time), 0],
                                        [-sin(first_scan_time), cos(first_scan_time), 0],
                                        [0, 0, 1]]).dot(self.m_main_transmitter.m_innpara[0:3].T)
            second_plane_vector = array([[cos(second_scan_time), sin(second_scan_time), 0],
                                         [-sin(second_scan_time), cos(second_scan_time), 0],
                                         [0, 0, 1]]).dot(self.m_main_transmitter.m_innpara[4:7].T)
            line = cross(second_plane_vector, first_plane_vector)
            if line[0] * first_plane_vector[1] - line[1] * first_plane_vector[0] < 0:
                line[0] = -line[0]
                line[1] = -line[1]
                line[2] = -line[2]
            hor_angle_sin = line[1] / sqrt(line[0] * line[0] + line[1] * line[1])
            hor_angle_cos = line[0] / sqrt(line[0] * line[0] + line[1] * line[1])
            if hor_angle_sin >= 0:
                if hor_angle_cos >= 0:
                    hor_angle[i] = arcsin(hor_angle_sin)
                else:
                    hor_angle[i] = pi - arcsin(hor_angle_sin)
            else:
                if hor_angle_cos >= 0:
                    hor_angle[i] = arcsin(hor_angle_sin) + pi * 2
                else:
                    hor_angle[i] = pi - arcsin(hor_angle_sin)
            # if line[1] > 0:
            #     if line[0] > 0:
            #         hor_angle[i] = arctan(line[1] / line[0])
            #     else:
            #         hor_angle[i] = arctan(line[1] / line[0]) + pi
            # else:
            #     if line[0] > 0:
            #         hor_angle[i] = arctan(line[1] / line[0]) + 2 * pi
            #     else:
            #         hor_angle[i] = arctan(line[1] / line[0]) + pi
            ver_angle[i] = arctan(line[2] / sqrt(line[0] * line[0] + line[1] * line[1]))
            if i % 2 == 1:
                point_length = self.m_ruler_length / abs(tan(ver_angle[i - 1]) - tan(ver_angle[i]))
                first_point = array([[point_length * cos(hor_angle[i - 1])],
                                     [point_length * sin(hor_angle[i - 1])],
                                     [point_length * tan(ver_angle[i - 1])]])
                self.m_main_station_point = append(self.m_main_station_point, first_point)
                second_point = array([[point_length * cos(hor_angle[i])],
                                      [point_length * sin(hor_angle[i])],
                                      [point_length * tan(ver_angle[i])]])
                self.m_main_station_point = append(self.m_main_station_point, second_point)
        self.m_main_station_point = self.m_main_station_point.reshape(-1, 3)

    def cal_sub_station_init_value(self):
        """
        Calculate the init value for sub_station(ruler point coordinate)
        :return:
        """
        hor_angle = zeros(self.m_point_number)
        ver_angle = zeros(self.m_point_number)
        for i in range(self.m_point_number):
            first_scan_time = self.m_sub_transmitter.m_ruler_calib_scan_time[i, 0] * 2 * pi
            second_scan_time = self.m_sub_transmitter.m_ruler_calib_scan_time[i, 1] * 2 * pi
            first_plane_vector = array([[cos(first_scan_time), sin(first_scan_time), 0],
                                        [-sin(first_scan_time), cos(first_scan_time), 0],
                                        [0, 0, 1]]).dot(self.m_sub_transmitter.m_innpara[0:3].T)
            second_plane_vector = array([[cos(second_scan_time), sin(second_scan_time), 0],
                                         [-sin(second_scan_time), cos(second_scan_time), 0],
                                         [0, 0, 1]]).dot(self.m_sub_transmitter.m_innpara[4:7].T)
            line = cross(second_plane_vector, first_plane_vector)
            if line[0] * first_plane_vector[1] - line[1] * first_plane_vector[0] < 0:
                line[0] = -line[0]
                line[1] = -line[1]
                line[2] = -line[2]
            hor_angle_sin = line[1] / sqrt(line[0] * line[0] + line[1] * line[1])
            hor_angle_cos = line[0] / sqrt(line[0] * line[0] + line[1] * line[1])
            if hor_angle_sin >= 0:
                if hor_angle_cos >= 0:
                    hor_angle[i] = arcsin(hor_angle_sin)
                else:
                    hor_angle[i] = pi - arcsin(hor_angle_sin)
            else:
                if hor_angle_cos >= 0:
                    hor_angle[i] = arcsin(hor_angle_sin) + pi * 2
                else:
                    hor_angle[i] = pi - arcsin(hor_angle_sin)
            # if line[1] > 0:
            #     if line[0] > 0:
            #         hor_angle[i] = arctan(line[1] / line[0])
            #     else:
            #         hor_angle[i] = arctan(line[1] / line[0]) + pi
            # else:
            #     if line[0] > 0:
            #         hor_angle[i] = arctan(line[1] / line[0]) + 2 * pi
            #     else:
            #         hor_angle[i] = arctan(line[1] / line[0]) + pi
            ver_angle[i] = arctan(line[2] / sqrt(line[0] * line[0] + line[1] * line[1]))
            if i % 2 == 1:
                point_length = self.m_ruler_length / abs(tan(ver_angle[i - 1]) - tan(ver_angle[i]))
                first_point = array([[point_length * cos(hor_angle[i - 1])],
                                     [point_length * sin(hor_angle[i - 1])],
                                     [point_length * tan(ver_angle[i - 1])]])
                self.m_sub_station_point = append(self.m_sub_station_point, first_point)
                second_point = array([[point_length * cos(hor_angle[i])],
                                      [point_length * sin(hor_angle[i])],
                                      [point_length * tan(ver_angle[i])]])
                self.m_sub_station_point = append(self.m_sub_station_point, second_point)
        self.m_sub_station_point = self.m_sub_station_point.reshape(-1, 3)
