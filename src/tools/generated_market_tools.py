from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.tools import market_tools


GENERATED_TOOL_FUNCTIONS = {}
GENERATED_WRITE_TOOL_NAMES = set()
GENERATED_CHAT_WORKSPACE_TOOL_NAMES = set()


def register_generated_tool(name, write=False, chat_workspace=True):
    def decorator(function):
        GENERATED_TOOL_FUNCTIONS[name] = function
        if write:
            GENERATED_WRITE_TOOL_NAMES.add(name)
        if chat_workspace:
            GENERATED_CHAT_WORKSPACE_TOOL_NAMES.add(name)
        return function

    return decorator


# <generated-tools>
# </generated-tools>
