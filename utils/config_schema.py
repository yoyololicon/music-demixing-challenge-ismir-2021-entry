from typing import Mapping, Any
import json
import os

dir = os.path.dirname(__file__)
CONFIG_SCHEMA: Mapping[str, Any] = json.load(
    open(os.path.join(dir, "schema.json")))
