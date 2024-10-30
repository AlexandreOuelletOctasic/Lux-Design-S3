from lux.utils import direction_to
import numpy as np
from state import State


class Agent:
    def __init__(self, player: str, env_cfg) -> None:
        self._state: State = State(player, env_cfg)
        self._policy = default_policy
        self._value_functions = None
        self._env_model = None

        np.random.seed(0)

    def value_func(self):
        pass

    def model(self):
        pass

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """implement this function to decide what actions to send to each available unit.

        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
        self._state.update(obs, step)
        actions = self._policy(self._state)
        return actions


unit_explore_locations = dict()
relic_node_positions = []
discovered_relic_nodes_ids = set()


def default_policy(state: State):
    unit_mask = np.array(state.obs["units_mask"]
                         [state.team_id])  # shape (max_units, )
    unit_positions = np.array(
        state.obs["units"]["position"][state.team_id]
    )  # shape (max_units, 2)
    # shape (max_units, 1)
    unit_energys = np.array(state.obs["units"]["energy"][state.team_id])
    observed_relic_node_positions = np.array(
        state.obs["relic_nodes"]
    )  # shape (max_relic_nodes, 2)
    observed_relic_nodes_mask = np.array(
        state.obs["relic_nodes_mask"]
    )  # shape (max_relic_nodes, )
    # points of each team, team_points[self.team_id] is the points of the your team
    team_points = np.array(state.obs["team_points"])

    # visible relic nodes
    visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])

    # basic strategy here is simply to have some units randomly explore and some units collecting as much energy as possible
    # and once a relic node is found, we send all units to move randomly around the first relic node to gain points
    # and information about where relic nodes are found are saved for the next match

    # save any new relic nodes that we discover for the rest of the game.
    for id in visible_relic_node_ids:
        if id not in discovered_relic_nodes_ids:
            discovered_relic_nodes_ids.add(id)
            relic_node_positions.append(observed_relic_node_positions[id])

    actions = np.zeros((state.env_cfg["max_units"], 3), dtype=int)

    # ids of units you can control at this timestep
    available_unit_ids = np.where(unit_mask)[0]

    # unit ids range from 0 to max_units - 1
    for unit_id in available_unit_ids:
        unit_pos = unit_positions[unit_id]
        unit_energy = unit_energys[unit_id]
        if len(relic_node_positions) > 0:
            nearest_relic_node_position = relic_node_positions[0]
            manhattan_distance = abs(
                unit_pos[0] - nearest_relic_node_position[0]
            ) + abs(unit_pos[1] - nearest_relic_node_position[1])

            # if close to the relic node we want to hover around it and hope to gain points
            if manhattan_distance <= 4:
                random_direction = np.random.randint(0, 5)
                actions[unit_id] = [random_direction, 0, 0]
            else:
                # otherwise we want to move towards the relic node
                actions[unit_id] = [
                    direction_to(unit_pos, nearest_relic_node_position),
                    0,
                    0,
                ]
        else:
            # randomly explore by picking a random location on the map and moving there for about 20 steps
            if state.step % 20 == 0 or unit_id not in unit_explore_locations:
                rand_loc = (
                    np.random.randint(0, state.env_cfg["map_width"]),
                    np.random.randint(0, state.env_cfg["map_height"]),
                )
                unit_explore_locations[unit_id] = rand_loc
            actions[unit_id] = [
                direction_to(unit_pos, unit_explore_locations[unit_id]),
                0,
                0,
            ]
            actions[unit_id] = [5, 2, -1]
    return actions
