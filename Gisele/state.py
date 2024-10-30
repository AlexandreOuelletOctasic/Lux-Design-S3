class State:
    def __init__(self, player: str, env_cfg):
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg

        self.previous_obs = None
        self.obs = None
        self.step: int = None

    def update(self, obs, step):
        self.previous_obs = self.obs
        self.obs = obs
        self.step = step
