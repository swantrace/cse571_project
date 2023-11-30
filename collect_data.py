# ************** STUDENTS EDIT THIS FILE **************
from SteeringBehaviors import Wander
import SimulationEnvironment as sim

import numpy as np


def collect_training_data(total_actions):
    # set-up environment
    sim_env = sim.SimulationEnvironment()

    # robot control
    action_repeat = 100
    steering_behavior = Wander(action_repeat)

    num_params = 7
    # STUDENTS: network_params will be used to store your training data
    # a single sample will be comprised of: sensor_readings, action, collision
    network_params = np.zeros((total_actions, num_params))

    for action_i in range(total_actions):
        progress = 100 * float(action_i) / total_actions
        print(f"Collecting Training Data {progress}%   ", end="\r", flush=True)

        # steering_force is used for robot control only
        action, steering_force = steering_behavior.get_action(
            action_i, sim_env.robot.body.angle
        )

        for action_timestep in range(action_repeat):
            if action_timestep == 0:
                _, collision, sensor_readings = sim_env.step(steering_force)
            else:
                _, collision, _ = sim_env.step(steering_force)

            if collision:
                steering_behavior.reset_action()
                # STUDENTS NOTE: this statement only EDITS collision of PREVIOUS action
                # if current action is very new.
                if (
                    action_timestep < action_repeat * 0.3 and action_timestep > 1
                ):  # in case prior action caused collision
                    network_params[action_timestep - 1][
                        -1
                    ] = collision  # share collision result with prior action
                break

        # STUDENTS: Update network_params.
        network_params[action_i][0:5] = sensor_readings
        network_params[action_i][5] = action
        network_params[action_i][6] = collision

    # STUDENTS: Save .csv here. Remember rows are individual samples, the first 5
    # columns are sensor values, the 6th is the action, and the 7th is collision.
    # Do not title the columns. Your .csv should look like the provided sample.
    # first five columns are float values, last two are integer
    data = np.array(network_params)
    formatted_data = np.vectorize(custom_format)(data)
    np.savetxt("submission.csv", formatted_data, fmt="%s", delimiter=",")
    # close environment
    sim_env.close()


def custom_format(x):
    if x == int(x):
        return f"{int(x)}"
    else:
        return f"{x:.12f}"


if __name__ == "__main__":
    total_actions = 20000
    collect_training_data(total_actions)
