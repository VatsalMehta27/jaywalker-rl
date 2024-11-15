import numpy as np

from utils.epsilon_schedules.epsilon_schedule import EpsilonSchedule


class ExponentialSchedule(EpsilonSchedule):
    def __init__(self, start_value, end_value, duration):
        super().__init__(start_value, end_value, duration)

        self.a = self.start_value
        self.b = (1 / (self.duration - 1)) * np.log(self.end_value / self.start_value)

    def get_value(self, step: int) -> float:
        """Return exponentially interpolated value between `value_from` and `value_to`interpolated value between.

        Returns {
            `value_from`, if step == 0 or less
            `value_to`, if step == num_steps - 1 or more
            the exponential interpolation between `value_from` and `value_to`, if 0 <= steps < num_steps
        }

        :param step: The step at which to compute the interpolation
        :rtype: Float. The interpolated value
        """
        if step <= 0:
            return self.start_value

        if step >= self.duration - 1:
            return self.end_value

        return self.a * np.exp(self.b * step)
