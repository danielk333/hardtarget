import datetime as dt

dt_format = "%Y-%m-%dT%H:%M:%S.%f"


def str_to_dt(dt_str: str):
    return dt.datetime.strptime(dt_str, dt_format).replace(tzinfo=dt.timezone.utc)
