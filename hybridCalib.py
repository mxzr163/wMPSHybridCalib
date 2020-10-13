from RestrictionEquation import *
from InitValueCalc import *
from wMPSalgorithm import *
import sys


def load_txt():
    if np.size(sys.argv) != 1:
        file = open(sys.argv[1])
    else:
        file = open("./data2.txt")
    data = []
    for line in file.readlines():
        data.append(line)
    return data


def solve_raw_data(data):
    line_number = np.size(data)
    solved_data = np.zeros((line_number, np.size(data[0].split())))
    for i in range(line_number):
        solved_data[i, :] = np.array(data[i].split(), np.float64)
    return solved_data


def handle_data(data):
    main_innpara = np.array(data[0].split(), np.float64)  # 发射站内参
    sub_innpara = np.array(data[1].split(), np.float64)
    ruler_length = np.float(data[2])
    ruler_number = np.int(data[3])
    ruler_scantime = data[4:4 + ruler_number * 2]
    calib_point_number = np.int(data[4 + ruler_number * 2])
    calib_point = solve_raw_data(data[5 + ruler_number * 2:5 + ruler_number * 2 + calib_point_number])
    calib_scantime = solve_raw_data(data[5 + ruler_number * 2 + calib_point_number:])
    return main_innpara, sub_innpara, ruler_length, ruler_number, ruler_scantime, \
        calib_point_number, calib_point, calib_scantime


def separate_scantime(scantime):
    scantime_number = np.size(scantime, 0)
    main_station_point_scantime = np.zeros((int(scantime_number / 2), 2))
    sub_station_point_scantime = np.zeros((int(scantime_number / 2), 2))
    for i in range(scantime_number):
        if i % 2 == 0:
            main_station_point_scantime[int(i / 2), :] = scantime[i, :]
        else:
            sub_station_point_scantime[int(i / 2), :] = scantime[i, :]
    return main_station_point_scantime, sub_station_point_scantime


def main():
    main_innpara, sub_innpara, ruler_length, ruler_number, ruler_scantime, calib_point_number, \
        control_point, calib_scantime = handle_data(load_txt())
    # 基准尺扫描时间
    main_station_point_scantime, sub_station_point_scantime = separate_scantime(calib_scantime)

    # 初始化发射站
    main_transmitter = Transmitter(main_innpara)
    sub_transmitter = Transmitter(sub_innpara)

    # 第一次迭代初值计算
    a = InitValueCalc(ruler_length, ruler_scantime, main_transmitter, sub_transmitter)
    rotation, transform = \
        calculate_transformations(a.m_sub_station_point, a.m_main_station_point)  # R * LeftPara + T = RightPara
    init_value = np.hstack((rotation, np.transpose(np.array([transform])), a.m_main_station_point.T))

    # 第一次迭代
    b = RestrictionEquation(main_transmitter, sub_transmitter, ruler_length, init_value)
    result = b.resolve_function()
    print("First Iter Complete!\n")
    print("Residual :\n", result.fun)
    sub_transmitter.m_rotation = result.x[0:9].reshape(3, 3).T
    sub_transmitter.m_transformation = result.x[9:12]  # 更新第二发射站外参

    main_transmitter.m_point_measure_scantime = main_station_point_scantime
    sub_transmitter.m_point_measure_scantime = sub_station_point_scantime
    coordinate_first_iter = calculate_coordinate(main_transmitter, sub_transmitter)  # 计算控制点在当前坐标系下坐标
    coordinate_first_iter_init = calculate_coordinate_with_ruler_time(main_transmitter, sub_transmitter)
    rotation_to_global, transform_to_global = calculate_transformations(coordinate_first_iter,
                                                                        control_point)  # 计算当前坐标系与全局坐标系转站关系
    main_transmitter.transform(rotation_to_global, transform_to_global)
    sub_transmitter.transform(rotation_to_global, transform_to_global)  # to global system
    # main_transmitter.m_rotation = np.array([[0.527477637200955,-0.849559223462635,0.004058088523245],
    #                                         [0.848773114486117 ,0.527184230187628,0.040755215212471],
    #                                         [-0.036763329262120,-0.018053068189155,0.999160920147762]])
    # main_transmitter.m_transformation = np.array([-2505.15644194695, -542.810242081196, -53.6473860256820])
    # sub_transmitter.m_rotation = np.array([[-0.698241866639994, 0.715671312683760, -0.0165187128009871],
    #                                        [-0.715431700920287, -0.698436311500665, -0.0185526304240255],
    #                                        [-0.0248148542087592, -0.00113621250212959, 0.999691418404573]])
    # sub_transmitter.m_transformation=np.array([1141.50291483433, 1784.81902295304,-31.8624080082071])

    second_iter_init_global_point = transform_point_inv(coordinate_first_iter_init, rotation_to_global,
                                                        transform_to_global)  # 基准尺在全局坐标系下坐标
    main_transmitter.m_point_calib_scantime = main_station_point_scantime
    sub_transmitter.m_point_calib_scantime = sub_station_point_scantime
    # second_iter_init_global_point = calculate_coordinate_with_ruler_time(main_transmitter, sub_transmitter)
    # 拼接初值
    main_matrix = np.hstack((main_transmitter.m_rotation, np.transpose([main_transmitter.m_transformation])))
    sub_matrix = np.hstack((sub_transmitter.m_rotation, np.transpose([sub_transmitter.m_transformation])))
    rt_matrix = np.hstack((main_matrix, sub_matrix))
    hybird_init_value = np.hstack((rt_matrix, second_iter_init_global_point.T))

    # 第二次迭代
    c = HybridRestrictionEquation(main_transmitter, sub_transmitter, ruler_length,
                                  control_point, hybird_init_value)
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
    pass


if __name__ == '__main__':
    main()
