import yaml
import aiofiles
from pathlib import Path

_ROOT = Path(__file__).absolute().parent


async def get_config(config: dict):
    # This loads things either ALL from configurable, or
    # all from the config.yaml
    # This is done intentionally to enforce an "all or nothing" configuration
    if "email" in config["configurable"]:
        return config["configurable"]
    else:
        async with aiofiles.open(_ROOT.joinpath("config.yaml"), mode='r') as stream:
            content = await stream.read()
            return yaml.safe_load(content)
