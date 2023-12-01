# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

@dataclass
class aim_config:
    experiment: str = "llama-recipes"
    # 'repo' can point to a locally accessible directory (e.g., '~/.aim') or a remote repository hosted on a server.
    # When 'remote_server_ip' or 'remote_server_port' is set, it designates a remote aim repo.
    # Otherwise, 'repo' specifies the directory, with a default of None representing '.aim'.
    repo: str = None
    remote_server_ip: str = None
    remote_server_port: int = None