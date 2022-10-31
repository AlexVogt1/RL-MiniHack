class Config:

    def __init__(self):
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        self.discount = 0.999
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25
        self.num_simulations = 400
        self.reuse_tree = False
        self.temperature = 0
        self.do_roll_outs = False
        self.number_of_roll_outs = 5
        self.max_roll_out_depth = 20
        self.do_roll_out_steps_with_simulation_true = False

class MCTSContinuousAgentConfig(Config):

    def __init__(self):
        super(MCTSContinuousAgentConfig, self).__init__()
        # single progressive widening
        self.C = 1
        self.alpha = 0.25

class Node:
    def __init__(self) -> None:
        self.visit_count = 0
        self.to_play = -1
        self.prior = 1      # uniform prior
        self.value_sum = 0
        self.children = {}
        self.observation = None
        self.reward = 0
        self.done = False
        self.game = None


    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0 or self.done:
            return 0
        return self.value_sum / self.visit_count
    

class MCTS:

    def __init__(self, conf, node):
        self.conf = conf
        self.node = node