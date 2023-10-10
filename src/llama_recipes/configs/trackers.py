# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

@dataclass
class aim_config:
    experiment: str = "llama-recipes"
    # This is either the location of a locallly accessible directory example `.run`
    # or can be the location of a remote repository which is hosted on a server.
    # if remote_server_ip or remote_server_port is set then the repo will be set to
    # remote aim repo, else the directory specified by `repo` takes precedence.
    # The directory defaults to None which means `.run` in aim.
    repo: str = None
    remote_server_ip: str = None
    remote_server_port: int = None