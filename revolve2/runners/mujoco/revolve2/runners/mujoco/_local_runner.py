import concurrent.futures
import math
import os
import tempfile
from typing import List, Optional

import cv2
import mujoco
import mujoco_viewer
import numpy as np
import numpy.typing as npt
from decimal import *



try:
    import logging

    old_len = len(logging.root.handlers)

    from dm_control import mjcf

    new_len = len(logging.root.handlers)

    assert (
        old_len + 1 == new_len
    ), "dm_control not adding logging handler as expected. Maybe they fixed their annoying behaviour? https://github.com/deepmind/dm_control/issues/314https://github.com/deepmind/dm_control/issues/314"

    logging.root.removeHandler(logging.root.handlers[-1])
except Exception as e:
    print("Failed to fix absl logging bug", e)
    pass

from pyrr import Quaternion, Vector3
from revolve2.core.physics.actor.urdf import to_urdf as physbot_to_urdf
from revolve2.core.physics.running import (
    ActorControl,
    ActorState,
    Batch,
    BatchResults,
    Environment,
    EnvironmentResults,
    EnvironmentState,
    RecordSettings,
    Runner,
)

MAX_POWER = 20
MIN_POWER = 0
STEP_SIZE = 0

class LocalRunner(Runner):
    """Runner for simulating using Mujoco."""

    _headless: bool
    _start_paused: bool
    _num_simulators: int

    def __init__(
        self,
        headless: bool = False,
        start_paused: bool = False,
        num_simulators: int = 1,
    ):
        """
        Initialize this object.

        :param headless: If True, the simulation will not be rendered. This drastically improves performance.
        :param start_paused: If True, start the simulation paused. Only possible when not in headless mode.
        :param num_simulators: The number of simulators to deploy in parallel. They will take one core each but will share space on the main python thread for calculating control.
        """
        assert (
            headless or num_simulators == 1
        ), "Cannot have parallel simulators when visualizing."

        assert not (
            headless and start_paused
        ), "Cannot start simulation paused in headless mode."

        self._headless = headless
        self._start_paused = start_paused
        self._num_simulators = num_simulators

    @classmethod
    def _run_environment(
        cls,
        env_index: int,
        env_descr: Environment,
        headless: bool,
        record_settings: Optional[RecordSettings],
        start_paused: bool,
        control_step: float,
        sample_step: float,
        simulation_time: int,
    ) -> EnvironmentResults:
        logging.info(f"Environment {env_index}")
        getcontext().prec = 40

        model = mujoco.MjModel.from_xml_string(cls._make_mjcf(env_descr))

        # TODO initial dof state
        data = mujoco.MjData(model)

        initial_targets = [
            dof_state
            for posed_actor in env_descr.actors
            for dof_state in posed_actor.dof_states
        ]
        cls._set_dof_targets(data, initial_targets)

        for posed_actor in env_descr.actors:
            posed_actor.dof_states

        if not headless or record_settings is not None:
            viewer = mujoco_viewer.MujocoViewer(
                model,
                data,
            )
            viewer._render_every_frame = False  # Private but functionality is not exposed and for now it breaks nothing.
            viewer._paused = start_paused

        if record_settings is not None:
            video_step = 1 / record_settings.fps
            video_file_path = f"{record_settings.video_directory}/{env_index}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video = cv2.VideoWriter(
                video_file_path,
                fourcc,
                record_settings.fps,
                (viewer.viewport.width, viewer.viewport.height),
            )

            viewer._hide_menu = True

        last_control_time = 0.0
        last_sample_time = 0.0
        last_video_time = 0.0  # time at which last video frame was saved
        remaining_power = Decimal(MAX_POWER)
        low_energy_threshold = MIN_POWER

        results = EnvironmentResults([])

        # sample initial state
        # results.environment_states.append(
        #     EnvironmentState(0.0, cls._get_actor_states(env_descr, data, model, remaining_power))
        # )

        while (time := data.time) < simulation_time:
            # do control if it is time
            if time >= last_control_time + control_step:
                last_control_time = math.floor(time / control_step) * control_step
                control_user = ActorControl()
                env_descr.controller.control(control_step, control_user)
                actor_targets = control_user._dof_targets
                actor_targets.sort(key=lambda t: t[0])
                targets = [
                    target
                    for actor_target in actor_targets
                    for target in actor_target[1]
                ]
                cls._set_dof_targets(data, targets)

            # sample state if it is time
            if time >= last_sample_time + sample_step:
                last_sample_time = int(time / sample_step) * sample_step

                actor_state = cls._get_actor_states(env_descr, data, model, remaining_power)

                if remaining_power <= low_energy_threshold:
                    remaining_power = 0
                    break

                # finish simulation if remaining power is non positive
                results.environment_states.append(
                    EnvironmentState(
                        time, actor_state
                    )
                )

            last_step_time = data.time

            # step simulation
            mujoco.mj_step(model, data)

            cur_step_time = data.time

            global STEP_SIZE
            STEP_SIZE = cur_step_time - last_step_time

            actor_state = cls._get_actor_states(env_descr, data, model, remaining_power)

            # update remaining power of robot
            remaining_power = actor_state[0].remaining_power

            # finish simulation if remaining power is non positive
            if remaining_power <= low_energy_threshold:
                remaining_power = 0
                break

            # render if not headless. also render when recording and if it time for a new video frame.
            if not headless or (
                record_settings is not None and time >= last_video_time + video_step
            ):
                viewer.render()

            # capture video frame if it's time
            if record_settings is not None and time >= last_video_time + video_step:
                last_video_time = int(time / video_step) * video_step

                # https://github.com/deepmind/mujoco/issues/285 (see also record.cc)
                img: npt.NDArray[np.uint8] = np.empty(
                    (viewer.viewport.height, viewer.viewport.width, 3),
                    dtype=np.uint8,
                )

                mujoco.mjr_readPixels(
                    rgb=img,
                    depth=None,
                    viewport=viewer.viewport,
                    con=viewer.ctx,
                )
                img = np.flip(img, axis=0)  # img is upside down initially
                video.write(img)

        if not headless or record_settings is not None:
            viewer.close()

        if record_settings is not None:
            video.release()

        # sample one final time
        # results.environment_states.append(
        #     EnvironmentState(time, cls._get_actor_states(env_descr, data, model, remaining_power))
        # )

        # print("FIRST STATE", "\n\n\n")
        # print(results.environment_states[0].time_seconds, results.environment_states[0].actor_states[0].remaining_power, results.environment_states[0].actor_states[0].njnts)
        # print("LAST STATE", "\n\n\n")
        # print(results.environment_states[-1].time_seconds, results.environment_states[-1].actor_states[0].remaining_power,
        #       results.environment_states[-1].actor_states[0].njnts)

        #print("Fitness remaining power: ",results.environment_states[-1].actor_states[0].remaining_power)
        #distance = float (math.sqrt((results.environment_states[0].actor_states[0].position[0] - results.environment_states[-1].actor_states[0].position[0]) ** 2 +
         #                           ((results.environment_states[0].actor_states[0].position[1] - results.environment_states[-1].actor_states[0].position[1]) ** 2)))
        #print("Fitness distance: ", distance)

        return results

    async def run_batch(
        self, batch: Batch, record_settings: Optional[RecordSettings] = None
    ) -> BatchResults:
        """
        Run the provided batch by simulating each contained environment.

        :param batch: The batch to run.
        :param record_settings: Optional settings for recording the runnings. If None, no recording is made.
        :returns: List of simulation states in ascending order of time.
        """
        logging.info("Starting simulation batch with mujoco.")

        control_step = 1 / batch.control_frequency
        sample_step = 1 / batch.sampling_frequency

        if record_settings is not None:
            os.makedirs(record_settings.video_directory, exist_ok=False)

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self._num_simulators
        ) as executor:
            futures = []
            for env_index, env_descr in enumerate(batch.environments):
                futures.append(executor.submit(
                    self._run_environment,
                    env_index,
                    env_descr,
                    self._headless,
                    record_settings,
                    self._start_paused,
                    control_step,
                    sample_step,
                    batch.simulation_time,
                ))
                '''for actor_index, posed_actor in enumerate(env_descr.actors):
                    print(actor_index, posed_actor.actor)'''


            results = BatchResults([future.result() for future in futures])

        logging.info("Finished batch.")

        return results

    @staticmethod
    def _make_mjcf(env_descr: Environment) -> str:
        env_mjcf = mjcf.RootElement(model="environment")

        env_mjcf.compiler.angle = "radian"

        env_mjcf.option.timestep = 0.0005
        env_mjcf.option.integrator = "RK4"

        env_mjcf.option.gravity = [0, 0, -9.81]

        env_mjcf.worldbody.add(
            "geom",
            name="ground",
            type="plane",
            size=[10, 10, 1],
            rgba=[0.2, 0.2, 0.2, 1],
        )
        env_mjcf.worldbody.add(
            "light",
            pos=[0, 0, 100],
            ambient=[0.5, 0.5, 0.5],
            directional=True,
            castshadow=False,
        )

        env_mjcf.visual.headlight.active = 0

        for actor_index, posed_actor in enumerate(env_descr.actors):
            urdf = physbot_to_urdf(
                posed_actor.actor,
                f"robot_{actor_index}",
                Vector3(),
                Quaternion(),
            )

            model = mujoco.MjModel.from_xml_string(urdf)

            # mujoco can only save to a file, not directly to string,
            # so we create a temporary file.
            with tempfile.NamedTemporaryFile(
                mode="r+", delete=True, suffix="_mujoco.urdf"
            ) as botfile:
                mujoco.mj_saveLastXML(botfile.name, model)
                robot = mjcf.from_file(botfile)

            force_range = 4.0  # limits force of each actuator (preventing jumping)

            for i, joint in enumerate(posed_actor.actor.joints):

                #     # Add damping to mimic a frictional joint.
                #     # robot.find(namespace="joint", identifier=joint.name).damping = 0.1
                #     # Add stiffness to mimic a spring joint.
                #Add armature to mimic the effect of a motor.

                robot.find(namespace="joint", identifier=joint.name).armature = 0.2
                #robot.find(namespace="joint", identifier=joint.name).stiffness = -0.1
                robot.actuator.add(
                    "position",
                    kp=150.0,
                    #kp = 5.0
                    ctrlrange="-1.0 1.0", # limits the range of the position controller
                    forcerange=f"{-force_range} {force_range}",  # limits the force of the position controller
                    joint=robot.find(
                        namespace="joint",
                        identifier=joint.name,
                    ),
                )
                robot.actuator.add(
                    "velocity",
                    kv=0.2,
                    #kv = 0.05,
                    ctrlrange="-1.0 1.0",  # limits the range of the position controller
                    forcerange=f"{-force_range} {force_range}",  # limits the force of the position controller
                    joint=robot.find(namespace="joint", identifier=joint.name),
                )

                bodies=robot.find(namespace="body", identifier=joint.name)
                bodies.add('site', name = f"IMU{i}")
                robot.sensor.add(
                    "torque",
                    site=f"IMU{i}"
                )

            robot.visual.headlight.active = 0
            attachment_frame = env_mjcf.attach(robot)
            attachment_frame.add("freejoint")
            attachment_frame.pos = [
                posed_actor.position.x,
                posed_actor.position.y,
                posed_actor.position.z,
            ]

            attachment_frame.quat = [
                posed_actor.orientation.x,
                posed_actor.orientation.y,
                posed_actor.orientation.z,
                posed_actor.orientation.w,
            ]

        xml = env_mjcf.to_xml_string()

        if not isinstance(xml, str):
            raise RuntimeError("Error generating mjcf xml.")

        return xml

    @classmethod
    def _get_actor_states(
        cls, env_descr: Environment, data: mujoco.MjData, model: mujoco.MjModel, remaining_power
    ) -> List[ActorState]:
        return [
            cls._get_actor_state(i, data, model, remaining_power) for i in range(len(env_descr.actors))
        ]

    @staticmethod
    def _get_actor_state(
        robot_index: int, data: mujoco.MjData, model: mujoco.MjModel, remaining_power: Decimal
    ) -> ActorState:
        bodyid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"robot_{robot_index}/",  # the slash is added by dm_control. ugly but deal with it
        )
        assert bodyid >= 0

        qindex = model.body_jntadr[bodyid]
        assert qindex >= 0
        # explicitly copy because the Vector3 and Quaternion classes don't copy the underlying structure
        position = Vector3([n for n in data.qpos[qindex : qindex + 3]])
        orientation = Quaternion([n for n in data.qpos[qindex + 3 : qindex + 3 + 4]])

        num_jnts = model.njnt - 1
        # print("Number of joints: ", num_jnts)

        jnt_hinge_indices = [i for i, j in enumerate(model.jnt_type) if j == mujoco.mjtJoint.mjJNT_HINGE]
        hinge_angles      = [data.qpos[model.jnt_qposadr[i]] for i in jnt_hinge_indices]
        hinge_vels        = [data.qvel[model.jnt_dofadr[i]] for i in jnt_hinge_indices]

        sensor_indices    = [i for i, j in enumerate(model.sensor_type) if j == mujoco.mjtSensor.mjSENS_TORQUE]
        hinge_torques     = [data.sensordata[model.sensor_adr[i]] for i in sensor_indices]

        # compute estimated power of the actor
        est_power = Decimal(0)
        for vel, angle, torque in zip(hinge_vels, hinge_angles, hinge_torques):
            est_power += Decimal(abs(torque)) * Decimal(abs(vel))

            # print(STEP_SIZE)
        # print("Power used: ", est_power)
        # print("Energy used: ", Decimal(STEP_SIZE) * est_power)

        remaining_power -= (est_power * Decimal(STEP_SIZE))

        # print("Number of joints: ", num_jnts)
        # print("Hinge angles: ", hinge_angles)
        # print("Hinge velocities: ", hinge_vels)
        # print("Hinge torques: ", hinge_torques)

        return ActorState(position, orientation, hinge_angles, hinge_vels, hinge_torques, remaining_power, num_jnts)

    @staticmethod
    def _set_dof_targets(data: mujoco.MjData, targets: List[float]) -> None:
        if len(targets) * 2 != len(data.ctrl):
            raise RuntimeError("Need to set a target for every dof")
        for i, target in enumerate(targets):
            data.ctrl[2 * i] = target
            data.ctrl[2 * i + 1] = 0
