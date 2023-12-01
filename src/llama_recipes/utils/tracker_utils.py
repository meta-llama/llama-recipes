import os
import importlib
from llama_recipes.configs import aim_config

class Tracker:
    def initialize(self, tracker_config):
        pass

    def load_params(self, params, context):
        pass

    def track(self, metric, name, stage):
        pass

class Aim(Tracker):
    run: None
    config: None

    def initialize(self, tracker_config):
        self.config = tracker_config
        assert isinstance(tracker_config, aim_config), f"config passed to AIM tracker is unknown: {type(tracker_config)}"

        try:
            AIM = importlib.import_module("aim")
            Run = AIM.Run
        except Exception:
            raise ImportError("Failed to import modules from pkg 'aim'.\n"
                              "Please run 'pip install aim==3.17.5' to install Aim before proceeding.")

        exp = self.config.experiment
        repo = self.config.repo

        if (self.config.remote_server_ip is not None) and (self.config.remote_server_port is not None):
            repo = f"aim://{self.config.remote_server_ip}:{self.config.remote_server_port}/"
        try:
            if repo is not None:
                print("Aimstack trying to connect to the repo "+repo)
                self.run = Run(repo=repo, experiment=exp)
            else:
                print("Aimstack using the default repo `.run`")
                self.run = Run(experiment=exp)
        except Exception:
            print("Failed to start Aim stack tracker")

    def load_params(self, params, context):
        run_params = {}
        for k, v in params.items():
            run_params[k] = v
        self.run[context] = run_params

    def track(self, metric, name, stage):
        if self.run is not None:
            self.run.track(metric, name=name, context={'subset':stage})

def get_tracker_by_name(name):
    if name is None:
        return None
    elif name == "aim":
        return Aim()
    else :
        print("Unknown Tracker "+str(name)+" Please check the name of the tracker passed. ")
        return None