import threading
import time
from typing import Any
import bosdyn.client
from bosdyn.client.math_helpers import SE3Pose
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder, blocking_stand, blocking_sit, blocking_selfright, block_until_arm_arrives, blocking_command
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.power import PowerClient
from bosdyn.client.gripper_camera_param import GripperCameraParamClient
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
from bundle_fetch.lib.spot_sdk.python.examples.estop.estop_gui import build_and_run_app as estop_gui_main
from types import SimpleNamespace
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.api import geometry_pb2
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b


class Spot(object):
    def __init__(self) -> None:
        # Create robot
        self.sdk = bosdyn.client.create_standard_sdk('bundle_fetch')
        self.robot = self.sdk.create_robot('192.168.80.3')
        bosdyn.client.util.authenticate(self.robot)
        self.robot.sync_with_directory()
        self.robot.time_sync.wait_for_sync()

        # Create clients
        self.lease_client = self.robot.ensure_client(LeaseClient.default_service_name)
        self.lease = self.lease_client.take()
        self.lease_keep_alive = bosdyn.client.lease.LeaseKeepAlive(self.lease_client)
        self.command_client: RobotCommandClient = self.robot.ensure_client(RobotCommandClient.default_service_name)
        self.robot_state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
        self.power_client: PowerClient = self.robot.ensure_client(PowerClient.default_service_name)
        self.gripper_param_client = self.robot.ensure_client(GripperCameraParamClient.default_service_name)

        # Estop robot
        self.estop_client = self.robot.ensure_client(EstopClient.default_service_name)
        self.estop_endpoint = EstopEndpoint(client=self.estop_client, name='my_estop', estop_timeout=9.0)
        self.estop_endpoint.force_simple_setup()
        self.estop_keep_alive = EstopKeepAlive(self.estop_endpoint)
        # def start_estop_thread():
        #     estop_gui_main('192.168.80.3', self.estop_client, SimpleNamespace(timeout=5, on_top=True, start_minimized=False))
        # self.estop_thread = threading.Thread(target=start_estop_thread)
        # self.estop_thread.start()

        # Power on robot
        print('powering on')
        self.robot.power_on(timeout_sec=20)
        print('powered on')

        # Self right robot
        print('self righting')
        blocking_selfright(self.command_client)
        print('self righted')

        # Stand robot
        print('standing up')
        # stand_command = RobotCommandBuilder.synchro_stand_command()
        # blocking_command(self.command_client, stand_command)
        blocking_stand(self.command_client)
        print('stood up')

        # Move arm
        print('moving arm')
        blocking_move_arm(self.robot_state_client, self.command_client, 0.5, 0, 0.5, 0, 0, 0, 1)
        print('moved arm')
        time.sleep(1)

        # Ready arm
        print('readying arm')
        ready_command = RobotCommandBuilder.arm_ready_command()
        ready_command_id = self.command_client.robot_command(lease=None, command=ready_command, end_time_secs=time.time() + 20)
        block_until_arm_arrives(self.command_client, ready_command_id, 10)
        print('unstowed arm')

        # Open gripper
        print('opening gripper')
        open_gripper = RobotCommandBuilder.claw_gripper_open_fraction_command(1)
        open_gripper_command_id = self.command_client.robot_command(open_gripper)
        time.sleep(1)
        print('opened gripper')

        # Close gripper
        print('closing gripper')
        close_gripper = RobotCommandBuilder.claw_gripper_open_fraction_command(0)
        close_gripper_command_id = self.command_client.robot_command(close_gripper)
        time.sleep(1)
        print('closed gripper')

        # Stow arm
        print('stowing arm')
        stow_command = RobotCommandBuilder.arm_stow_command()
        stow_command_id = self.command_client.robot_command(lease=None, command=stow_command, end_time_secs=time.time() + 20)
        block_until_arm_arrives(self.command_client, stow_command_id, 10)
        print('stowed arm')

        # Move robot forward
        print('moving forward')
        blocking_move_body(self.robot, self.command_client, -0.1, 0, 0)
        print('moved forward')

        # Sit robot
        print('sitting down')
        blocking_sit(self.command_client)
        print('sat down')

        # Power off robot
        print('powering off')
        self.robot.power_off(timeout_sec=20)
        print('powered off')


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