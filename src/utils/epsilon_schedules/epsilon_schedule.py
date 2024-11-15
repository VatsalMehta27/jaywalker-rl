from abc import ABC, abstractmethod


class EpsilonSchedule(ABC):
    def __init__(self, start_value, end_value, duration):
        self.start_value = start_value
        self.end_value = end_value
        self.duration = duration

    @abstractmethod
    def get_value(self, step: int) -> float:
        pass
