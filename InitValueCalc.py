from wMPSalgorithm import *


class InitValueCalc:
    def __init__(self, ruler_length, raw_scantime, main_transmitter, sub_transmitter):
        self.m_ruler_length = ruler_length
        self.m_raw_scantime = raw_scantime
        self.m_point_number = int(np.size(self.m_raw_scantime) / 2)
        self.m_main_transmitter = main_transmitter
        self.m_sub_transmitter = sub_transmitter
        self.m_main_station_point = np.array([])
        self.m_sub_station_point = np.array([])
        self.handle_raw_scantime()
        self.cal_main_station_init_value()
        self.cal_sub_station_init_value()

    def handle_raw_scantime(self):
        for i in range(self.m_point_number * 2):
            if i % 2:
                self.m_sub_transmitter.ruler_calib_scantime_append(np.array(self.m_raw_scantime[i].split(),
                                                                            np.float64))
            else:
                self.m_main_transmitter.ruler_calib_scantime_append(np.array(self.m_raw_scantime[i].split(),
                                                                             np.float64))

    def cal_main_station_init_value(self):
        hor_angle = np.zeros(self.m_point_number)
        ver_angle = np.zeros(self.m_point_number)
        for i in range(self.m_point_number):
            first_scantime = self.m_main_transmitter.m_ruler_calib_scantime[i, 0] * 2 * np.pi
            second_scantime = self.m_main_transmitter.m_ruler_calib_scantime[i, 1] * 2 * np.pi
            first_plane_vector = np.array([[np.cos(first_scantime), np.sin(first_scantime), 0],
                                           [-np.sin(first_scantime), np.cos(first_scantime), 0],
                                           [0, 0, 1]]).dot(self.m_main_transmitter.m_innpara[0:3].T)
            second_plane_vector = np.array([[np.cos(second_scantime), np.sin(second_scantime), 0],
                                            [-np.sin(second_scantime), np.cos(second_scantime), 0],
                                            [0, 0, 1]]).dot(self.m_main_transmitter.m_innpara[4:7].T)
            line = np.cross(second_plane_vector, first_plane_vector)
            if line[0]*first_plane_vector[1]-line[1]*first_plane_vector[0]<0:
                line[0] = -line[0]
                line[1] = -line[1]
                line[2] = -line[2]
            hor_angle_sin = line[1]/np.sqrt(line[0]*line[0] + line[1] * line[1])
            hor_angle_cos = line[0]/np.sqrt(line[0]*line[0] + line[1] * line[1])
            if hor_angle_sin >= 0:
                if hor_angle_cos >= 0:
                    hor_angle[i] = np.arcsin(hor_angle_sin)
                else:
                    hor_angle[i] = np.pi - np.arcsin(hor_angle_sin)
            else:
                if hor_angle_cos >= 0:
                    hor_angle[i] = np.arcsin(hor_angle_sin) +np.pi * 2
                else:
                    hor_angle[i] = np.pi - np.arcsin(hor_angle_sin)
            # if line[1] > 0:
            #     if line[0] > 0:
            #         hor_angle[i] = np.arctan(line[1] / line[0])
            #     else:
            #         hor_angle[i] = np.arctan(line[1] / line[0]) + np.pi
            # else:
            #     if line[0] > 0:
            #         hor_angle[i] = np.arctan(line[1] / line[0]) + 2 * np.pi
            #     else:
            #         hor_angle[i] = np.arctan(line[1] / line[0]) + np.pi
            ver_angle[i] = np.arctan(line[2] / np.sqrt(line[0] * line[0] + line[1] * line[1]))
            if i % 2 == 1:
                point_length = self.m_ruler_length / np.abs(np.tan(ver_angle[i - 1]) - np.tan(ver_angle[i]))
                first_point = np.array([[point_length * np.cos(hor_angle[i - 1])],
                                        [point_length * np.sin(hor_angle[i - 1])],
                                        [point_length * np.tan(ver_angle[i - 1])]])
                self.m_main_station_point = np.append(self.m_main_station_point, first_point)
                second_point = np.array([[point_length * np.cos(hor_angle[i])],
                                         [point_length * np.sin(hor_angle[i])],
                                         [point_length * np.tan(ver_angle[i])]])
                self.m_main_station_point = np.append(self.m_main_station_point, second_point)
        self.m_main_station_point = self.m_main_station_point.reshape(-1, 3)

    def cal_sub_station_init_value(self):
        hor_angle = np.zeros(self.m_point_number)
        ver_angle = np.zeros(self.m_point_number)
        for i in range(self.m_point_number):
            first_scantime = self.m_sub_transmitter.m_ruler_calib_scantime[i, 0] * 2 * np.pi
            second_scantime = self.m_sub_transmitter.m_ruler_calib_scantime[i, 1] * 2 * np.pi
            first_plane_vector = np.array([[np.cos(first_scantime), np.sin(first_scantime), 0],
                                           [-np.sin(first_scantime), np.cos(first_scantime), 0],
                                           [0, 0, 1]]).dot(self.m_sub_transmitter.m_innpara[0:3].T)
            second_plane_vector = np.array([[np.cos(second_scantime), np.sin(second_scantime), 0],
                                            [-np.sin(second_scantime), np.cos(second_scantime), 0],
                                            [0, 0, 1]]).dot(self.m_sub_transmitter.m_innpara[4:7].T)
            line = np.cross(second_plane_vector, first_plane_vector)
            if line[0]*first_plane_vector[1]-line[1]*first_plane_vector[0]<0:
                line[0] = -line[0]
                line[1] = -line[1]
                line[2] = -line[2]
            hor_angle_sin = line[1]/np.sqrt(line[0]*line[0] + line[1] * line[1])
            hor_angle_cos = line[0]/np.sqrt(line[0]*line[0] + line[1] * line[1])
            if hor_angle_sin >= 0:
                if hor_angle_cos >= 0:
                    hor_angle[i] = np.arcsin(hor_angle_sin)
                else:
                    hor_angle[i] = np.pi - np.arcsin(hor_angle_sin)
            else:
                if hor_angle_cos >= 0:
                    hor_angle[i] = np.arcsin(hor_angle_sin) +np.pi * 2
                else:
                    hor_angle[i] = np.pi - np.arcsin(hor_angle_sin)
            # if line[1] > 0:
            #     if line[0] > 0:
            #         hor_angle[i] = np.arctan(line[1] / line[0])
            #     else:
            #         hor_angle[i] = np.arctan(line[1] / line[0]) + np.pi
            # else:
            #     if line[0] > 0:
            #         hor_angle[i] = np.arctan(line[1] / line[0]) + 2 * np.pi
            #     else:
            #         hor_angle[i] = np.arctan(line[1] / line[0]) + np.pi
            ver_angle[i] = np.arctan(line[2] / np.sqrt(line[0] * line[0] + line[1] * line[1]))
            if i % 2 == 1:
                point_length = self.m_ruler_length / np.abs(np.tan(ver_angle[i - 1]) - np.tan(ver_angle[i]))
                first_point = np.array([[point_length * np.cos(hor_angle[i - 1])],
                                        [point_length * np.sin(hor_angle[i - 1])],
                                        [point_length * np.tan(ver_angle[i - 1])]])
                self.m_sub_station_point = np.append(self.m_sub_station_point, first_point)
                second_point = np.array([[point_length * np.cos(hor_angle[i])],
                                         [point_length * np.sin(hor_angle[i])],
                                         [point_length * np.tan(ver_angle[i])]])
                self.m_sub_station_point = np.append(self.m_sub_station_point, second_point)
        self.m_sub_station_point = self.m_sub_station_point.reshape(-1, 3)
