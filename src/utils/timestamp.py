import datetime
import time


def get_current_timestamp() -> str:
    return datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
