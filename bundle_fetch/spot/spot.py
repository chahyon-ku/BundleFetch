import threading
import time
from typing import Any
import bosdyn.client
from bosdyn.client.math_helpers import SE3Pose
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder, blocking_stand, blocking_sit, blocking_selfright, block_until_arm_arrives, blocking_command
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.power import PowerClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.gripper_camera_param import GripperCameraParamClient
from bosdyn.client.image import ImageClient
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
# from bundle_fetch.lib.spot_sdk.python.examples.estop.estop_gui import build_and_run_app as estop_gui_main
from types import SimpleNamespace
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.api import geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b
from bosdyn.api import gripper_camera_param_pb2
from google.protobuf import wrappers_pb2
import math
# import torch.multiprocessing as mp


class Spot(object):
    def __init__(self, spot_queue) -> None:
        self.spot_queue = spot_queue
        # def start_estop_thread():
        #     estop_gui_main('192.168.80.3', estop_client, SimpleNamespace(timeout=5, on_top=True, start_minimized=False))
        # estop_thread = threading.Thread(target=start_estop_thread)
        # estop_thread.start()

        # Ready arm
        # print('readying arm')
        # ready_command = RobotCommandBuilder.arm_ready_command()
        # ready_command_id = command_client.robot_command(lease=None, command=ready_command, end_time_secs=time.time() + 20)
        # block_until_arm_arrives(command_client, ready_command_id, 10)
        # print('unstowed arm')

        # # Close gripper
        # print('closing gripper')
        # close_gripper = RobotCommandBuilder.claw_gripper_open_fraction_command(0)
        # close_gripper_command_id = command_client.robot_command(close_gripper)
        # time.sleep(1)
        # print('closed gripper')

        # # Stow arm
        # print('stowing arm')
        # stow_command = RobotCommandBuilder.arm_stow_command()
        # stow_command_id = command_client.robot_command(lease=None, command=stow_command, end_time_secs=time.time() + 20)
        # block_until_arm_arrives(command_client, stow_command_id, 10)
        # print('stowed arm')

        # # Move robot forward
        # print('moving forward')
        # blocking_move_body(robot, command_client, -0.1, 0, 0)
        # print('moved forward')

        # # Sit robot
        # print('sitting down')
        # blocking_sit(command_client)
        # print('sat down')

        # # Power off robot
        # print('powering off')
        # robot.power_off(timeout_sec=20)
        # print('powered off')

    def __call__(self) -> None:
        print('[SPOT] spot process started')
        # Create robot
        sdk = bosdyn.client.create_standard_sdk('bundle_fetch')
        # robot = sdk.create_robot('192.168.80.3')
        robot = sdk.create_robot('10.0.0.3')
        bosdyn.client.util.authenticate(robot)
        robot.sync_with_directory()
        robot.time_sync.wait_for_sync(5)
        print('[SPOT] created robot')

        # Create clients
        lease_client = robot.ensure_client(LeaseClient.default_service_name)
        lease = lease_client.take()
        lease_keep_alive = bosdyn.client.lease.LeaseKeepAlive(lease_client)
        command_client: RobotCommandClient = robot.ensure_client(RobotCommandClient.default_service_name)
        robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
        power_client: PowerClient = robot.ensure_client(PowerClient.default_service_name)
        gripper_param_client = robot.ensure_client(GripperCameraParamClient.default_service_name)
        manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)
        estop_client = robot.ensure_client(EstopClient.default_service_name)
        estop_endpoint = EstopEndpoint(client=estop_client, name='my_estop', estop_timeout=9.0)
        estop_endpoint.force_simple_setup()
        estop_keep_alive = EstopKeepAlive(estop_endpoint)
        image_client = robot.ensure_client(ImageClient.default_service_name)
        print('[SPOT] created clients')

        # Robot ready
        robot.power_on(timeout_sec=20)
        blocking_selfright(command_client)
        blocking_stand(command_client)
        blocking_move_arm(robot_state_client, command_client, 0.5, 0, 0.5, 0, 0.25, 0, 0.968)
        open_gripper = RobotCommandBuilder.claw_gripper_open_fraction_command(1)
        open_gripper_command_id = command_client.robot_command(open_gripper)
        time.sleep(1)
        print('[SPOT] robot ready')
        image_sources = ['hand_color_image', 'hand_depth_in_hand_color_frame']

        phase = 0
        while True:
            # set light based on sign wave
            phase += 0.1
            # time.sleep(0.1)
            try:
                command = self.spot_queue.get(timeout=0.1)
                print('[SPOT] COMMAND RECEIVED')
                blocking_set_light(gripper_param_client, command)
            except:
                command = None
            # if command is None:
            #     continue

            # image_responses = image_client.get_image_from_sources(['hand_color_image'])
            # image = image_responses[0]
            # src_xy, dst_xy = command
            # src_xy = geometry_pb2.Vec2(x=wrappers_pb2.FloatValue(src_xy[1]), y=wrappers_pb2.FloatValue(src_xy[0]))
            # # Build the proto
            # walk_to = manipulation_api_pb2.WalkToObjectInImage(
            #     pixel_xy=src_xy, transforms_snapshot_for_camera=image.shot.transforms_snapshot,
            #     frame_name_image_sensor=image.shot.frame_name_image_sensor,
            #     camera_model=image.source.pinhole, offset_distance=0.5)

            # # Ask the robot to pick up the object
            # walk_to_request = manipulation_api_pb2.ManipulationApiRequest(
            #     walk_to_object_in_image=walk_to)

            # # Send the request
            # cmd_response = manipulation_api_client.manipulation_api_command(
            #     manipulation_api_request=walk_to_request)

            # # Get feedback from the robot
            # while True:
            #     time.sleep(0.25)
            #     feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
            #         manipulation_cmd_id=cmd_response.manipulation_cmd_id)

            #     # Send the request
            #     response = manipulation_api_client.manipulation_api_feedback_command(
            #         manipulation_api_feedback_request=feedback_request)

            #     print('Current state: ',
            #         manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state))

            #     if response.current_state == manipulation_api_pb2.MANIP_STATE_DONE:
            #         break


def blocking_move_body(robot, command_client, x, y, yaw, stairs=False):
    frame_tree_snapshot = robot.get_frame_tree_snapshot()
    move_command = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
        x,
        y,
        yaw,
        frame_tree_snapshot, # 'odom',
        RobotCommandBuilder.mobility_params(stair_hint=False)
    )
    command_id = command_client.robot_command(
        lease=None,
        command=move_command,
        end_time_secs=time.time() + 20
    )
    while True:
        feedback = command_client.robot_command_feedback(command_id)
        mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
        if mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
            print('Failed to reach the goal')
            break
        traj_feedback = mobility_feedback.se2_trajectory_feedback
        if (traj_feedback.status == traj_feedback.STATUS_AT_GOAL and
                traj_feedback.body_movement_status == traj_feedback.BODY_STATUS_SETTLED):
            print('Arrived at the goal.')
            break
        time.sleep(0.1)

def blocking_move_arm(robot_state_client, command_client, x, y, z, qx, qy, qz, qw):
    flat_body_t_hand = geometry_pb2.Vec3(x=x, y=y, z=z)
    flat_body_Q_hand = geometry_pb2.Quaternion(w=qw, x=qx, y=qy, z=qz)
    flat_body_T_hand = geometry_pb2.SE3Pose(position=flat_body_t_hand,
                                            rotation=flat_body_Q_hand)

    robot_state = robot_state_client.get_robot_state()
    odom_T_flat_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                        ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)
    odom_T_hand = odom_T_flat_body * SE3Pose.from_proto(flat_body_T_hand)
    seconds = 2

    arm_command = RobotCommandBuilder.arm_pose_command(
        odom_T_hand.x, odom_T_hand.y, odom_T_hand.z, odom_T_hand.rot.w, odom_T_hand.rot.x,
        odom_T_hand.rot.y, odom_T_hand.rot.z, ODOM_FRAME_NAME, seconds)
    
    arm_command_id = command_client.robot_command(arm_command)
    command_client.robot_command_feedback(arm_command_id)
    block_until_arm_arrives(command_client, arm_command_id, seconds)


def blocking_set_light(gripper_param_client: GripperCameraParamClient, fraction):
    if fraction == 0:
        led_mode = gripper_camera_param_pb2.GripperCameraParams.LED_MODE_OFF
        led_torch_brightness = None
    else:
        led_mode = gripper_camera_param_pb2.GripperCameraParams.LED_MODE_TORCH
        led_torch_brightness = wrappers_pb2.FloatValue(value=fraction)
    params = gripper_camera_param_pb2.GripperCameraParams(
        led_mode=led_mode, led_torch_brightness=led_torch_brightness)

    request = gripper_camera_param_pb2.GripperCameraParamRequest(params=params)
    gripper_param_client.set_camera_params(request)
