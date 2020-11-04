#
#  wMPSAlgorithm.py
#  wMPSHybridCalib
#
#  Created by Zhang Rao on 2020/11/3.
#  Copyright © 2020 Zhang Rao. All rights reserved.
#
from numpy import eye, size, zeros, array, vstack, mean, tile, \
    sum, cos, sin, pi, arctan2, sqrt
from numpy.linalg import svd, inv, det


class Transmitter:
    """
        This class is used to Save the transmitter information.

        Attributes:
            m_innpara:  A (8,) array for Transmitter inner parameter.
            m_rotation: A (3, 3) array for Transmitter's Rotation matrix.
            m_transformation: A (3, 1) array for Transmitter's Transformation matrix.
            m_ruler_calib_scan_time: A (ruler point number, 2) array for saving transmitter ruler scan time.
            m_point_calib_scan_time: A (control point number, 2) array for saving transmitter control point scan time.
            m_point_measure_scan_time: A (point number, 2) array for saving transmitter measurement scan time.
    """
    def __init__(self, innpara, rotation=eye(3), transformation=zeros(3),
                 ruler_calib_scan_time=array([]), point_calib_scan_time=array([]),
                 point_measure_scan_time=array([])):
        self.m_innpara = innpara
        self.m_rotation = rotation
        self.m_transformation = transformation
        self.m_ruler_calib_scan_time = ruler_calib_scan_time
        self.m_point_calib_scan_time = point_calib_scan_time
        self.m_point_measure_scan_time = point_measure_scan_time

    def ruler_calib_scan_time_append(self, scan_time):
        """
        Add scan_time to the end of m_ruler_calib_scan_time.
        :param scan_time: A (2, ) array for ruler scan time.
        :return:Null
        """
        if size(self.m_ruler_calib_scan_time) != 0:
            self.m_ruler_calib_scan_time = vstack((self.m_ruler_calib_scan_time, scan_time))
        else:
            self.m_ruler_calib_scan_time = array([scan_time])

    def transform(self, rotation, transformation):
        self.m_transformation = self.m_rotation.dot(transformation) + self.m_transformation
        self.m_rotation = self.m_rotation.dot(rotation)


def calculate_transformations(point_group_dst, point_group_ref):
    """
    Calculate the transformation from reference point group to destination point group.
    R * ref + T = dst
    :param point_group_dst: The Destination Point Group.
    :param point_group_ref: The Point Group need to transform.
    :return: Rotation, Transformation
    """
    point_number = size(point_group_dst, 0)
    mess_center_of_dst = mean(point_group_dst, 0)
    mess_center_of_ref = mean(point_group_ref, 0)
    d0 = point_group_dst - tile(mess_center_of_dst, (1, point_number)).reshape(-1, 3)
    d1 = point_group_ref - tile(mess_center_of_ref, (1, point_number)).reshape(-1, 3)
    k_matrix = zeros((3, 3))
    k_matrix[0, 0] = sum(d0[:, 0] * d1[:, 0])
    k_matrix[0, 1] = sum(d0[:, 0] * d1[:, 1])
    k_matrix[0, 2] = sum(d0[:, 0] * d1[:, 2])
    k_matrix[1, 0] = sum(d0[:, 1] * d1[:, 0])
    k_matrix[1, 1] = sum(d0[:, 1] * d1[:, 1])
    k_matrix[1, 2] = sum(d0[:, 1] * d1[:, 2])
    k_matrix[2, 0] = sum(d0[:, 2] * d1[:, 0])
    k_matrix[2, 1] = sum(d0[:, 2] * d1[:, 1])
    k_matrix[2, 2] = sum(d0[:, 2] * d1[:, 2])
    print("enter")
    u, s, v = svd(k_matrix)
    print("stop")
    rotation = u.dot(v)
    if -0.9999 > det(rotation) > -1.0001:
        v[2, :] = -v[2, :]
        rotation = u.dot(v)
    transformation = mess_center_of_dst - rotation.dot(mess_center_of_ref)
    return rotation, transformation


def calculate_coordinate(main_transmitter, sub_transmitter):
    """
    Using m_point_measure_scan_time to calculate coordinate
    :param main_transmitter: A class transmitter for main station
    :param sub_transmitter: A class transmitter for sub station
    :return: coordinate
    """
    point_number = size(main_transmitter.m_point_measure_scan_time, 0)
    coordinate = zeros((point_number, 3))
    for i in range(point_number):
        a = zeros((4, 3))
        b = zeros((4, 1))
        ct1 = cos(main_transmitter.m_point_measure_scan_time[i, 0] * 2 * pi)
        st1 = sin(main_transmitter.m_point_measure_scan_time[i, 0] * 2 * pi)
        a[0, :] = main_transmitter.m_innpara[0:3].dot([[ct1, -st1, 0],
                                                       [st1, ct1, 0],
                                                       [0, 0, 1]]).dot(main_transmitter.m_rotation)
        b[0] = -main_transmitter.m_innpara[3] - a[0, :].dot(main_transmitter.m_rotation.T) \
            .dot(main_transmitter.m_transformation)
        ct2 = cos(main_transmitter.m_point_measure_scan_time[i, 1] * 2 * pi)
        st2 = sin(main_transmitter.m_point_measure_scan_time[i, 1] * 2 * pi)
        a[1, :] = main_transmitter.m_innpara[4:7].dot([[ct2, -st2, 0],
                                                       [st2, ct2, 0],
                                                       [0, 0, 1]]).dot(main_transmitter.m_rotation)
        b[1] = -main_transmitter.m_innpara[7] - a[1, :].dot(main_transmitter.m_rotation.T) \
            .dot(main_transmitter.m_transformation)
        ct1 = cos(sub_transmitter.m_point_measure_scan_time[i, 0] * 2 * pi)
        st1 = sin(sub_transmitter.m_point_measure_scan_time[i, 0] * 2 * pi)
        a[2, :] = sub_transmitter.m_innpara[0:3].dot([[ct1, -st1, 0],
                                                      [st1, ct1, 0],
                                                      [0, 0, 1]]).dot(sub_transmitter.m_rotation)
        b[2] = -sub_transmitter.m_innpara[3] - a[2, :].dot(sub_transmitter.m_rotation.T) \
            .dot(sub_transmitter.m_transformation)
        ct2 = cos(sub_transmitter.m_point_measure_scan_time[i, 1] * 2 * pi)
        st2 = sin(sub_transmitter.m_point_measure_scan_time[i, 1] * 2 * pi)
        a[3, :] = sub_transmitter.m_innpara[4:7].dot([[ct2, -st2, 0],
                                                      [st2, ct2, 0],
                                                      [0, 0, 1]]).dot(sub_transmitter.m_rotation)
        b[3] = -sub_transmitter.m_innpara[7] - a[3, :].dot(sub_transmitter.m_rotation.T) \
            .dot(sub_transmitter.m_transformation)
        coordinate[i, :] = inv(a.T.dot(a)).dot(a.T.dot(b)).T
    return coordinate


def calculate_coordinate_with_ruler_time(main_transmitter, sub_transmitter):
    """
    Using m_ruler_calib_scan_time to calculate coordinate
    :param main_transmitter: A class transmitter for main station
    :param sub_transmitter: A class transmitter for sub station
    :return: coordinate
    """

    point_number = size(main_transmitter.m_ruler_calib_scan_time, 0)
    coordinate = zeros((point_number, 3))
    for i in range(point_number):
        a = zeros((4, 3))
        b = zeros((4, 1))
        ct1 = cos(main_transmitter.m_ruler_calib_scan_time[i, 0] * 2 * pi)
        st1 = sin(main_transmitter.m_ruler_calib_scan_time[i, 0] * 2 * pi)
        a[0, :] = main_transmitter.m_innpara[0:3].dot([[ct1, -st1, 0],
                                                       [st1, ct1, 0],
                                                       [0, 0, 1]]).dot(main_transmitter.m_rotation)
        b[0] = -main_transmitter.m_innpara[3] - a[0, :].dot(main_transmitter.m_rotation.T) \
            .dot(main_transmitter.m_transformation)
        ct2 = cos(main_transmitter.m_ruler_calib_scan_time[i, 1] * 2 * pi)
        st2 = sin(main_transmitter.m_ruler_calib_scan_time[i, 1] * 2 * pi)
        a[1, :] = main_transmitter.m_innpara[4:7].dot([[ct2, -st2, 0],
                                                       [st2, ct2, 0],
                                                       [0, 0, 1]]).dot(main_transmitter.m_rotation)
        b[1] = -main_transmitter.m_innpara[7] - a[1, :].dot(main_transmitter.m_rotation.T) \
            .dot(main_transmitter.m_transformation)
        ct1 = cos(sub_transmitter.m_ruler_calib_scan_time[i, 0] * 2 * pi)
        st1 = sin(sub_transmitter.m_ruler_calib_scan_time[i, 0] * 2 * pi)
        a[2, :] = sub_transmitter.m_innpara[0:3].dot([[ct1, -st1, 0],
                                                      [st1, ct1, 0],
                                                      [0, 0, 1]]).dot(sub_transmitter.m_rotation)
        b[2] = -sub_transmitter.m_innpara[3] - a[2, :].dot(sub_transmitter.m_rotation.T) \
            .dot(sub_transmitter.m_transformation)
        ct2 = cos(sub_transmitter.m_ruler_calib_scan_time[i, 1] * 2 * pi)
        st2 = sin(sub_transmitter.m_ruler_calib_scan_time[i, 1] * 2 * pi)
        a[3, :] = sub_transmitter.m_innpara[4:7].dot([[ct2, -st2, 0],
                                                      [st2, ct2, 0],
                                                      [0, 0, 1]]).dot(sub_transmitter.m_rotation)
        b[3] = -sub_transmitter.m_innpara[7] - a[3, :].dot(sub_transmitter.m_rotation.T) \
            .dot(sub_transmitter.m_transformation)
        coordinate[i, :] = inv(a.T.dot(a)).dot(a.T.dot(b)).T
    return coordinate


def calculate_coordinate_with_calib_time(main_transmitter, sub_transmitter):
    """
    Using m_point_calib_scan_time to calculate coordinate
    :param main_transmitter: A class transmitter for main station
    :param sub_transmitter: A class transmitter for sub station
    :return: coordinate
    """

    point_number = size(main_transmitter.m_point_calib_scan_time, 0)
    coordinate = zeros((point_number, 3))
    for i in range(point_number):
        a = zeros((4, 3))
        b = zeros((4, 1))
        ct1 = cos(main_transmitter.m_point_calib_scan_time[i, 0] * 2 * pi)
        st1 = sin(main_transmitter.m_point_calib_scan_time[i, 0] * 2 * pi)
        a[0, :] = main_transmitter.m_innpara[0:3].dot([[ct1, -st1, 0],
                                                       [st1, ct1, 0],
                                                       [0, 0, 1]]).dot(main_transmitter.m_rotation)
        b[0] = -main_transmitter.m_innpara[3] - a[0, :].dot(main_transmitter.m_rotation.T) \
            .dot(main_transmitter.m_transformation)
        ct2 = cos(main_transmitter.m_point_calib_scan_time[i, 1] * 2 * pi)
        st2 = sin(main_transmitter.m_point_calib_scan_time[i, 1] * 2 * pi)
        a[1, :] = main_transmitter.m_innpara[4:7].dot([[ct2, -st2, 0],
                                                       [st2, ct2, 0],
                                                       [0, 0, 1]]).dot(main_transmitter.m_rotation)
        b[1] = -main_transmitter.m_innpara[7] - a[1, :].dot(main_transmitter.m_rotation.T) \
            .dot(main_transmitter.m_transformation)
        ct1 = cos(sub_transmitter.m_point_calib_scan_time[i, 0] * 2 * pi)
        st1 = sin(sub_transmitter.m_point_calib_scan_time[i, 0] * 2 * pi)
        a[2, :] = sub_transmitter.m_innpara[0:3].dot([[ct1, -st1, 0],
                                                      [st1, ct1, 0],
                                                      [0, 0, 1]]).dot(sub_transmitter.m_rotation)
        b[2] = -sub_transmitter.m_innpara[3] - a[2, :].dot(sub_transmitter.m_rotation.T) \
            .dot(sub_transmitter.m_transformation)
        ct2 = cos(sub_transmitter.m_point_calib_scan_time[i, 1] * 2 * pi)
        st2 = sin(sub_transmitter.m_point_calib_scan_time[i, 1] * 2 * pi)
        a[3, :] = sub_transmitter.m_innpara[4:7].dot([[ct2, -st2, 0],
                                                      [st2, ct2, 0],
                                                      [0, 0, 1]]).dot(sub_transmitter.m_rotation)
        b[3] = -sub_transmitter.m_innpara[7] - a[3, :].dot(sub_transmitter.m_rotation.T) \
            .dot(sub_transmitter.m_transformation)
        coordinate[i, :] = inv(a.T.dot(a)).dot(a.T.dot(b)).T
    return coordinate


def transform_point(point_group, rotation, transformation):
    """
    transform point group by given rotation and transformation.
    R * X + T
    :param point_group: A (point number, 3) array for saving the point coordinate.
    :param rotation: A (3, 3) array for rotation matrix.
    :param transformation: A (3,) array for transformation.
    :return point_dst: A (point number, 3) array for saving the transformed point coordinate.
    """
    point_number = size(point_group, 0)
    point_dst = zeros((point_number, 3))
    for i in range(point_number):
        point_dst[i, :] = rotation.dot(point_group[i, :]) + transformation
    return point_dst


def transform_point_inv(point_group, rotation, transformation):
    """
    transform point group by given rotation and transformation.
    R * (X -T)
    :param point_group: A (point number, 3) array for saving the point coordinate.
    :param rotation: A (3, 3) array for rotation matrix.
    :param transformation: A (3,) array for transformation.
    :return point_dst: A (point number, 3) array for saving the transformed point coordinate.
    """
    point_number = size(point_group, 0)
    point_dst = zeros((point_number, 3))
    for i in range(point_number):
        point_dst[i, :] = rotation.T.dot(point_group[i, :] - transformation)
    return point_dst


def euler_to_rotation(angle_x, angle_y, angle_z):
    """
    generate rotation matrix by x, y ,z.
    R = Rz * Ry * Rx
    :param angle_x: A float for angle x.
    :param angle_y: A float for angle y.
    :param angle_z: A float for angle z.
    :return rotation: A (3, 3) matrix for rotation.
    """
    rotation_x = array([[1, 0, 0],
                        [0, cos(angle_x), -sin(angle_x)],
                        [0, sin(angle_x), cos(angle_x)]])
    rotation_y = array([[cos(angle_y), 0, sin(angle_y)],
                        [0, 1, 0],
                        [-sin(angle_y), 0, cos(angle_y)]])
    rotation_z = array([[cos(angle_z), -sin(angle_z), 0],
                        [sin(angle_z), cos(angle_z), 0],
                        [0, 0, 1]])
    rotation = rotation_z.dot(rotation_y.dot(rotation_x))
    return rotation


def rotation_to_euler(rotation):
    """
    generate rotation angle x, y, z  by rotation matrix.
    R = Rz * Ry * Rx
    :param rotation: A (3, 3) matrix for rotation.
    :return angle_x: A float for angle x.
    :return angle_y: A float for angle y.
    :return angle_z: A float for angle z.
    """
    angle_x = arctan2(rotation[2, 1], rotation[2, 2])
    angle_y = arctan2(-rotation[2, 0], sqrt(rotation[0, 0] * rotation[0, 0] + rotation[1, 0] * rotation[1, 0]))
    angle_z = arctan2(rotation[1, 0], rotation[0, 0])
    return angle_x, angle_y, angle_z
