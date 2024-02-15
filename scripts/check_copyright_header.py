import re
from pathlib import Path

WORK_DIR = Path(__file__).parents[1]
PATTERN = "(Meta Platforms, Inc. and affiliates)|(Facebook, Inc(\.|,)? and its affiliates)|([0-9]{4}-present(\.|,)? Facebook)|([0-9]{4}(\.|,)? Facebook)"

HEADER = """# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.\n\n"""

if __name__ == "__main__":
    for ext in ["*.py", "*.sh"]:
        for file in WORK_DIR.rglob(ext):
            text = file.read_text()
            if not re.search(PATTERN, text):
                text = HEADER + text
                file.write_text(text)
        