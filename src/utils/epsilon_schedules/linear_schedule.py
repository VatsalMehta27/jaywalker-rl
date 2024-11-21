from src.utils.epsilon_schedules.epsilon_schedule import EpsilonSchedule


class LinearSchedule(EpsilonSchedule):
    def __init__(self, start_value, end_value, duration):
        super().__init__(start_value, end_value, duration)
        # difference between the start value and the end value
        self._schedule_amount = end_value - start_value

    def get_value(self, step: int) -> float:
        # logic: if time > duration, use the end value, else use the scheduled value
        return self.start_value + self._schedule_amount * min(
            1.0, step * 1.0 / self.duration
        )
