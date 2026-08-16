"""
This file implements a wrapper for saving simulation states to disk.
This data collection wrapper is useful for collecting demonstrations.
"""

import json
import os
import time

import numpy as np

from robosuite.utils.mjcf_utils import save_sim_model
from robosuite.wrappers import Wrapper


class DataCollectionWrapper(Wrapper):
    def __init__(
        self,
        env,
        directory,
        collect_freq=1,
        flush_freq=100,
        use_env_xml_for_reset=False,
        record_joint_position_fields=False,
        joint_delta_scale=0.05,
        joint_position_observation_key="robot0_joint_pos",
    ):
        """
        Initializes the data collection wrapper.

        Args:
            env (MujocoEnv): The environment to monitor.
            directory (str): Where to store collected data.
            collect_freq (int): How often to save simulation state, in terms of environment steps.
            flush_freq (int): How frequently to dump data to disk, in terms of environment steps.
            use_env_xml_for_reset (bool): Whether to use the robosuite env XML string or the xml
                                          string from self.sim for resetting the environment.
            record_joint_position_fields (bool): Whether to save absolute joint targets, the
                                                 current joint-position observation, and normalized
                                                 joint-delta labels in every action info.
            joint_delta_scale (float): Joint delta in radians represented by a normalized delta of 1.
            joint_position_observation_key (str): Observation key used for current joint positions.
        """
        super().__init__(env)

        # the base directory for all logging
        self.directory = directory
        self.use_env_xml_for_reset = use_env_xml_for_reset
        self.record_joint_position_fields = record_joint_position_fields
        self.joint_delta_scale = float(joint_delta_scale)
        self.joint_position_observation_key = joint_position_observation_key
        if self.record_joint_position_fields and self.joint_delta_scale <= 0:
            raise ValueError("joint_delta_scale must be positive")

        # in-memory cache for simulation states and action info
        self.states = []
        self.action_infos = []  # stores information about actions taken
        self.successful = False  # stores success state of demonstration

        # how often to save simulation state, in terms of environment steps
        self.collect_freq = collect_freq

        # how frequently to dump data to disk, in terms of environment steps
        self.flush_freq = flush_freq

        if not os.path.exists(directory):
            print("DataCollectionWrapper: making new directory at {}".format(directory))
            os.makedirs(directory)

        # store logging directory for current episode
        self.ep_directory = None

        # remember whether any environment interaction has occurred
        self.has_interaction = False

        # some variables for remembering the current episode's initial state and model xml
        self._current_task_instance_state = None
        self._current_task_instance_xml = None

    def _start_new_episode(self):
        """
        Bookkeeping to do at the start of each new episode.
        """

        # flush any data left over from the previous episode if any interactions have happened
        if self.has_interaction:
            self._flush()

        # timesteps in current episode
        self.t = 0
        self.has_interaction = False

        # save the task instance (will be saved on the first env interaction)

        # NOTE: was originally set to self.env.model.get_xml()
        # That was causing the following issue in rare cases:
        # ValueError: Error: eigenvalues of mesh inertia violate A + B >= C
        # then, switched to self.env.sim.model.get_xml() which does not create this issue
        # however, that leads to subtle changes in the physics, such as fixture doors being harder to close
        # so, in order to address both issues, added an flag to choose between the two methods
        if self.use_env_xml_for_reset:
            self._current_task_instance_xml = self.env.model.get_xml()
        else:
            self._current_task_instance_xml = self.env.sim.model.get_xml()
        self._current_task_instance_state = np.array(self.env.sim.get_state().flatten())

        # trick for ensuring that we can play MuJoCo demonstrations back
        # deterministically by using the recorded actions open loop
        self.env.set_ep_meta(self.env.get_ep_meta())
        self.env.reset_from_xml_string(self._current_task_instance_xml)
        self.env.sim.reset()
        self.env.sim.set_state_from_flattened(self._current_task_instance_state)
        self.env.sim.forward()

    def _on_first_interaction(self):
        """
        Bookkeeping for first timestep of episode.
        This function is necessary to make sure that logging only happens after the first
        step call to the simulation, instead of on the reset (people tend to call
        reset more than is necessary in code).

        Raises:
            AssertionError: [Episode path already exists]
        """

        self.has_interaction = True

        # create a directory with a timestamp
        t1, t2 = str(time.time()).split(".")
        self.ep_directory = os.path.join(self.directory, "ep_{}_{}".format(t1, t2))
        assert not os.path.exists(self.ep_directory)
        print("DataCollectionWrapper: making folder at {}".format(self.ep_directory))
        os.makedirs(self.ep_directory)

        # save the model xml
        xml_path = os.path.join(self.ep_directory, "model.xml")
        with open(xml_path, "w") as f:
            f.write(self._current_task_instance_xml)

        # save the episode info to json file
        ep_meta_path = os.path.join(self.ep_directory, "ep_meta.json")
        with open(ep_meta_path, "w") as f:
            json.dump(self.env.get_ep_meta(), f)

        # save initial state and action
        assert len(self.states) == 0
        self.states.append(self._current_task_instance_state)

    def _flush(self):
        """
        Method to flush internal state to disk.
        """
        t1, t2 = str(time.time()).split(".")
        state_path = os.path.join(self.ep_directory, "state_{}_{}.npz".format(t1, t2))
        if hasattr(self.env, "unwrapped"):
            env_name = self.env.unwrapped.__class__.__name__
        else:
            env_name = self.env.__class__.__name__
        np.savez(
            state_path,
            states=np.array(self.states),
            action_infos=self.action_infos,
            successful=self.successful,
            env=env_name,
        )
        self.states = []
        self.action_infos = []
        self.successful = False

    def reset(self):
        """
        Extends vanilla reset() function call to accommodate data collection

        Returns:
            OrderedDict: Environment observation space after reset occurs
        """
        self.env.unset_ep_meta()  # unset any episode meta data that was previously set
        ret = super().reset()
        self._start_new_episode()
        return ret

    def step(self, action):
        """
        Extends vanilla step() function call to accommodate data collection

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (OrderedDict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        joint_fields = None
        if self.record_joint_position_fields:
            observation = self.env._get_observations(force_update=True)
            if self.joint_position_observation_key not in observation:
                raise KeyError(
                    f"Missing joint-position observation {self.joint_position_observation_key!r}; "
                    f"available keys are {sorted(observation)}"
                )
            joint_position = np.asarray(observation[self.joint_position_observation_key], dtype=float).copy()
            absolute_joint_target = np.asarray(action[: len(joint_position)], dtype=float).copy()
            joint_delta = absolute_joint_target - joint_position
            reference_scaled_joint_delta = joint_delta / self.joint_delta_scale
            joint_fields = {
                "robot0_joint_pos": joint_position,
                "joint_position": joint_position,
                "absolute_joint_target": absolute_joint_target,
                "joint_delta": joint_delta,
                "joint_delta_scale": self.joint_delta_scale,
                "joint_delta_reference_scaled": reference_scaled_joint_delta,
                "joint_delta_exceeds_reference_scale": bool(
                    np.any(np.abs(joint_delta) > self.joint_delta_scale)
                ),
                "actions_absolute_joint_position": np.asarray(action, dtype=float).copy(),
                "actions_joint_delta": np.concatenate(
                    [joint_delta, np.asarray(action[-1:], dtype=float)]
                ),
            }

        ret = super().step(action)
        self.t += 1

        # on the first time step, make directories for logging
        if not self.has_interaction:
            self._on_first_interaction()

        # collect the current simulation state if necessary
        if self.t % self.collect_freq == 0:
            state = self.env.sim.get_state().flatten()
            self.states.append(state)

            info = {}
            info["actions"] = np.array(action)

            # (if applicable) store absolute actions
            step_info = ret[3]
            if "action_abs" in step_info.keys():
                info["actions_abs"] = np.array(step_info["action_abs"])
            if joint_fields is not None:
                info.update(joint_fields)

            self.action_infos.append(info)

        # check if the demonstration is successful
        if self.env._check_success():
            self.successful = True

        # flush collected data to disk if necessary
        if self.t % self.flush_freq == 0:
            self._flush()

        return ret

    def close(self):
        """
        Override close method in order to flush left over data
        """
        if self.has_interaction:
            self._flush()
        self.env.close()
