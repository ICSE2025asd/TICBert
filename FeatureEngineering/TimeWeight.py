import math
import re
import time
from datetime import datetime

import numpy as np


def time_parser(time_str: str, convert_to: str):
    ISO8601 = "^(\d{4})-(0?[1-9]|1[0-2])-(0?[1-9]|[12][0-9]|3[01])(T|\s)(0?[0-9]|1[0-9]|2[0-4]):([0-5][0-9]):([0-5][0-9])(\.\d{1,3})((Z|[+-][0-9]{4})|(\s))$"
    Ymd = "%Y-%m-%d"
    BdY = "%B %d, %Y"
    BY = "%B %Y"
    Ym = "%Y-%m"
    t = datetime.replace(datetime.now(), year=1970, month=1, day=1, hour=0, minute=0, second=0)
    if re.match(ISO8601, time_str) and re.match(ISO8601, time_str).span()[1] == len(time_str):
        time_str = time_str.split("T")[0]
        t = datetime.strptime(time_str, Ymd)
    if convert_to == "Ymd":
        pass
    elif convert_to == "BdY":
        t = datetime.strftime(t, BdY)
    elif convert_to == "BY":
        t = datetime.strftime(t, BY)
    elif convert_to == "Ym":
        t = datetime.strftime(t, Ym)
    return t


def time2stamp(t: datetime):
    return time.mktime(t.timetuple())


def delta_month(t1: datetime, t2: datetime):
    delta = abs((t1.year - t2.year) * 12 + (t1.month - t2.month))
    return delta


def delta_year(t1: datetime, t2: datetime):
    delta = abs(t1.year - t2.year)
    return delta


class GaussWeight:
    max_date: datetime.date
    mid_date: datetime.date
    a: float
    b: float
    c: float

    def __init__(self, time_series: list, param_type: str, granularity: str):
        self.time_series = [time_parser(_, "Ymd") for _ in time_series]
        self.param_type = param_type
        self.granularity = granularity
        self.max_date = max(self.time_series)
        self.min_date = min(self.time_series)
        self.offset_sequence = np.array([self.time_encoder(_) for _ in time_series])
        self.sequence_max = np.max(self.offset_sequence)
        self.sequence_mean = np.mean(self.offset_sequence)
        self.sequence_std = np.std(self.offset_sequence)
        self.a = 0.5
        self.b = self.sequence_max
        if param_type == "emp":
            self.c = max(self.offset_sequence) ** 2 / (2 * math.log(2))
        elif param_type == "std":
            self.c = self.sequence_std ** 2
        self.d = 1 - self.a

    def time_encoder(self, time_str: str):
        t = time_parser(time_str, "Ymd")
        encoded_time = 0
        if self.granularity == "day":
            encoded_time = (t - self.min_date).days
        elif self.granularity == "month":
            encoded_time = delta_month(t, self.min_date)
        elif self.granularity == "year":
            encoded_time = delta_year(t, self.min_date)
        elif self.granularity == "timestamp":
            encoded_time = time2stamp(t)
        return encoded_time

    def get_weight(self, time_str: str):
        encoded_time = self.time_encoder(time_str)
        if encoded_time > self.b:
            return 1
        else:
            return self.a * math.pow(math.e, -(encoded_time - self.b) ** 2 / (2 * self.c)) + self.d
