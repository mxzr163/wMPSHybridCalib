#
#  InitValueCalc.py
#  wMPSHybridCalib
#
#  Main
#  pyinstaller lack libiomp5md.dll
#  Created by Zhang Rao on 2020/11/3.
#  Copyright © 2020 Zhang Rao. All rights reserved.
#
from RestrictionEquation import *
from InitValueCalc import *
from wMPSAlgorithm import *
from tkinter import filedialog
from numpy import size, zeros, array, float64, hstack, transpose


def load_txt():
    """
    load txt file from select path
    :return: a (line number,) array for raw string data
    """
    file = open(filedialog.askopenfilename())
    data = []
    for line in file.readlines():
        data.append(line)
    return data


def solve_raw_data(data):
    """
    split string data to array
    :param data: raw string data
    :return: a (line number, col number) array for raw string data
    """
    line_number = size(data)
    solved_data = zeros((line_number, size(data[0].split())))
    for i in range(line_number):
        solved_data[i, :] = array(data[i].split(), float64)
    return solved_data


def handle_data(data):
    """
    Handle the raw data
    data format:
    (, 8) main station inner parameter
    (, 8) sub station inner parameter
    (, 1) ruler length
    (, 1) ruler point number(2 * ruler position number)
    (2 * ruler point number, 2) ruler scan time
    (, 1) control point number
    (control point number, 3) control point coordinate
    (2 * control point number, 2) control point scan time
    :param data: raw data
    :return: main_innpara, sub_innpara, ruler_length, ruler_number, ruler_scan_time,
        calib_point_number, calib_point, calib_scan_time
    """
    main_innpara = array(data[0].split(), float64)  # 发射站内参
    sub_innpara = array(data[1].split(), float64)
    ruler_length = float(data[2])
    ruler_number = int(data[3])
    ruler_scan_time = data[4:4 + ruler_number * 2]
    calib_point_number = int(data[4 + ruler_number * 2])
    calib_point = solve_raw_data(data[5 + ruler_number * 2:5 + ruler_number * 2 + calib_point_number])
    calib_scan_time = solve_raw_data(data[5 + ruler_number * 2 + calib_point_number:])
    return main_innpara, sub_innpara, ruler_length, ruler_number, ruler_scan_time, \
        calib_point_number, calib_point, calib_scan_time


def separate_scan_time(scan_time):
    """
    separate point scan time to different transmitter
    :param scan_time: mixed scan time
    :return: main_station_point_scan_time, sub_station_point_scan_time
    """
    scan_time_number = size(scan_time, 0)
    main_station_point_scan_time = zeros((int(scan_time_number / 2), 2))
    sub_station_point_scan_time = zeros((int(scan_time_number / 2), 2))
    for i in range(scan_time_number):
        if i % 2 == 0:
            main_station_point_scan_time[int(i / 2), :] = scan_time[i, :]
        else:
            sub_station_point_scan_time[int(i / 2), :] = scan_time[i, :]
    return main_station_point_scan_time, sub_station_point_scan_time


def write_calib_result(main_transmitter, sub_transmitter):
    """
    write calibration results to selected path
    :param main_transmitter:
    :param sub_transmitter:
    :return:
    """
    file_name = filedialog.askdirectory() + "/CalibResult.txt"
    with open(file_name, "w") as f:
        angle_x, angle_y, angle_z = rotation_to_euler(main_transmitter.m_rotation)
        f.write("1 " + str(angle_x) + " " + str(angle_y) + " " + str(angle_z) + " " +
                str(main_transmitter.m_transformation[0]) + " " + str(main_transmitter.m_transformation[1])
                + " " + str(main_transmitter.m_transformation[2]) + "\n")
        angle_x, angle_y, angle_z = rotation_to_euler(sub_transmitter.m_rotation)
        f.write("2 " + str(angle_x) + " " + str(angle_y) + " " + str(angle_z) + " " +
                str(sub_transmitter.m_transformation[0]) + " " + str(sub_transmitter.m_transformation[1])
                + " " + str(sub_transmitter.m_transformation[2]) + "\n")


def main():
    """
    MAIN
    :return:
    """
    main_innpara, sub_innpara, ruler_length, ruler_number, ruler_scan_time, calib_point_number, \
        control_point, calib_scan_time = handle_data(load_txt())
    # 基准尺扫描时间
    main_station_point_scan_time, sub_station_point_scan_time = separate_scan_time(calib_scan_time)
    print("Separate_scan_time")
    # 初始化发射站
    main_transmitter = Transmitter(main_innpara)
    sub_transmitter = Transmitter(sub_innpara)
    print("Init Transmitter")
    # 第一次迭代初值计算
    a = InitValueCalc(ruler_length, ruler_scan_time, main_transmitter, sub_transmitter)
    print("InitValueCalc")

    global_rotation, global_transform = \
        calculate_transformations(a.m_sub_station_point, a.m_main_station_point)  # R * LeftPara + T = RightPara
    print("calculate_transformations")
    init_value = hstack((global_rotation, transpose(array([global_transform])), a.m_main_station_point.T))

    # 第一次迭代
    b = RestrictionEquation(main_transmitter, sub_transmitter, ruler_length, init_value)
    result = b.resolve_function()
    print("First Iter Complete!\n")
    print("Residual :\n", result.fun)
    sub_transmitter.m_rotation = result.x[0:9].reshape(3, 3).T
    sub_transmitter.m_transformation = result.x[9:12]  # 更新第二发射站外参

    main_transmitter.m_point_measure_scan_time = main_station_point_scan_time
    sub_transmitter.m_point_measure_scan_time = sub_station_point_scan_time
    coordinate_first_iter = calculate_coordinate(main_transmitter, sub_transmitter)  # 计算控制点在当前坐标系下坐标
    coordinate_first_iter_init = calculate_coordinate_with_ruler_time(main_transmitter, sub_transmitter)
    rotation_to_global, transform_to_global = calculate_transformations(coordinate_first_iter,
                                                                        control_point)  # 计算当前坐标系与全局坐标系转站关系
    main_transmitter.transform(rotation_to_global, transform_to_global)
    sub_transmitter.transform(rotation_to_global, transform_to_global)  # to global system
    # main_transmitter.m_rotation = array([[0.527477637200955,-0.849559223462635,0.004058088523245],
    #                                         [0.848773114486117 ,0.527184230187628,0.040755215212471],
    #                                         [-0.036763329262120,-0.018053068189155,0.999160920147762]])
    # main_transmitter.m_transformation = array([-2505.15644194695, -542.810242081196, -53.6473860256820])
    # sub_transmitter.m_rotation = array([[-0.698241866639994, 0.715671312683760, -0.0165187128009871],
    #                                        [-0.715431700920287, -0.698436311500665, -0.0185526304240255],
    #                                        [-0.0248148542087592, -0.00113621250212959, 0.999691418404573]])
    # sub_transmitter.m_transformation=array([1141.50291483433, 1784.81902295304,-31.8624080082071])

    second_iter_init_global_point = transform_point_inv(coordinate_first_iter_init, rotation_to_global,
                                                        transform_to_global)  # 基准尺在全局坐标系下坐标
    main_transmitter.m_point_calib_scan_time = main_station_point_scan_time
    sub_transmitter.m_point_calib_scan_time = sub_station_point_scan_time
    # second_iter_init_global_point = calculate_coordinate_with_ruler_time(main_transmitter, sub_transmitter)
    # 拼接初值
    main_matrix = hstack((main_transmitter.m_rotation, transpose([main_transmitter.m_transformation])))
    sub_matrix = hstack((sub_transmitter.m_rotation, transpose([sub_transmitter.m_transformation])))
    rt_matrix = hstack((main_matrix, sub_matrix))
    hybrid_init_value = hstack((rt_matrix, second_iter_init_global_point.T))

    # 第二次迭代
    c = HybridRestrictionEquation(main_transmitter, sub_transmitter, ruler_length,
                                  control_point, hybrid_init_value)
    result_second_iter = c.resolve_function()
    print("Second Iter Complete")
    print("Residual :\n", result_second_iter.fun)

    # 发射站外参
    main_transmitter.m_rotation = result_second_iter.x[0:9].reshape(3, 3).T
    main_transmitter.m_transformation = result_second_iter.x[9:12]
    sub_transmitter.m_rotation = result_second_iter.x[12:21].reshape(3, 3).T
    sub_transmitter.m_transformation = result_second_iter.x[21:24]

    # 第二次迭代坐标（基准尺）
    coordinate_second_iter = calculate_coordinate_with_ruler_time(main_transmitter, sub_transmitter)
    coordinate_second_iter_control_point = calculate_coordinate_with_calib_time(main_transmitter, sub_transmitter)

    print("Ruler point :\n", coordinate_second_iter)
    print("Control point :\n", coordinate_second_iter_control_point)
    print("Control point error :\n", control_point - coordinate_second_iter_control_point)
    write_calib_result(main_transmitter, sub_transmitter)
    pass


if __name__ == '__main__':
    main()
