from wMPSalgorithm import *

class InitValueCalc:
    def __init__(self, ruler_length, raw_scantime, main_transmitter, sub_transmitter):
        self.m_ruler_length = ruler_length
        self.m_raw_scantime = raw_scantime
        self.m_point_number = int(size(self.m_raw_scantime) / 2)
        self.m_main_transmitter = main_transmitter
        self.m_sub_transmitter = sub_transmitter
        self.m_main_station_point = array([])
        self.m_sub_station_point = array([])
        self.handle_raw_scantime()
        self.cal_main_station_init_value()
        self.cal_sub_station_init_value()

    def handle_raw_scantime(self):
        for i in range(self.m_point_number * 2):
            if i % 2:
                self.m_sub_transmitter.ruler_calib_scantime_append(array(self.m_raw_scantime[i].split(),
                                                                            float64))
            else:
                self.m_main_transmitter.ruler_calib_scantime_append(array(self.m_raw_scantime[i].split(),
                                                                             float64))

    def cal_main_station_init_value(self):
        hor_angle = zeros(self.m_point_number)
        ver_angle = zeros(self.m_point_number)
        for i in range(self.m_point_number):
            first_scantime = self.m_main_transmitter.m_ruler_calib_scantime[i, 0] * 2 * pi
            second_scantime = self.m_main_transmitter.m_ruler_calib_scantime[i, 1] * 2 * pi
            first_plane_vector = array([[cos(first_scantime), sin(first_scantime), 0],
                                           [-sin(first_scantime), cos(first_scantime), 0],
                                           [0, 0, 1]]).dot(self.m_main_transmitter.m_innpara[0:3].T)
            second_plane_vector = array([[cos(second_scantime), sin(second_scantime), 0],
                                            [-sin(second_scantime), cos(second_scantime), 0],
                                            [0, 0, 1]]).dot(self.m_main_transmitter.m_innpara[4:7].T)
            line = cross(second_plane_vector, first_plane_vector)
            if line[0]*first_plane_vector[1]-line[1]*first_plane_vector[0]<0:
                line[0] = -line[0]
                line[1] = -line[1]
                line[2] = -line[2]
            hor_angle_sin = line[1]/sqrt(line[0]*line[0] + line[1] * line[1])
            hor_angle_cos = line[0]/sqrt(line[0]*line[0] + line[1] * line[1])
            if hor_angle_sin >= 0:
                if hor_angle_cos >= 0:
                    hor_angle[i] = arcsin(hor_angle_sin)
                else:
                    hor_angle[i] = pi - arcsin(hor_angle_sin)
            else:
                if hor_angle_cos >= 0:
                    hor_angle[i] = arcsin(hor_angle_sin) +pi * 2
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
        hor_angle = zeros(self.m_point_number)
        ver_angle = zeros(self.m_point_number)
        for i in range(self.m_point_number):
            first_scantime = self.m_sub_transmitter.m_ruler_calib_scantime[i, 0] * 2 * pi
            second_scantime = self.m_sub_transmitter.m_ruler_calib_scantime[i, 1] * 2 * pi
            first_plane_vector = array([[cos(first_scantime), sin(first_scantime), 0],
                                           [-sin(first_scantime), cos(first_scantime), 0],
                                           [0, 0, 1]]).dot(self.m_sub_transmitter.m_innpara[0:3].T)
            second_plane_vector = array([[cos(second_scantime), sin(second_scantime), 0],
                                            [-sin(second_scantime), cos(second_scantime), 0],
                                            [0, 0, 1]]).dot(self.m_sub_transmitter.m_innpara[4:7].T)
            line = cross(second_plane_vector, first_plane_vector)
            if line[0]*first_plane_vector[1]-line[1]*first_plane_vector[0]<0:
                line[0] = -line[0]
                line[1] = -line[1]
                line[2] = -line[2]
            hor_angle_sin = line[1]/sqrt(line[0]*line[0] + line[1] * line[1])
            hor_angle_cos = line[0]/sqrt(line[0]*line[0] + line[1] * line[1])
            if hor_angle_sin >= 0:
                if hor_angle_cos >= 0:
                    hor_angle[i] = arcsin(hor_angle_sin)
                else:
                    hor_angle[i] = pi - arcsin(hor_angle_sin)
            else:
                if hor_angle_cos >= 0:
                    hor_angle[i] = arcsin(hor_angle_sin) +pi * 2
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
