import numpy as np


class ExponentialSchedule:
    def __init__(self, start_value, end_value, duration):
        """Exponential schedule from `value_from` to `value_to` in `num_steps` steps.

        $value(t) = a * exp(bt)$

        :param value_from: Initial value
        :param value_to: Final value
        :param num_steps: Number of steps for the exponential schedule
        """
        self.start_value = start_value
        self.end_value = end_value
        self.duration = duration

        self.a = self.start_value
        self.b = (1 / (self.duration - 1)) * np.log(self.end_value / self.start_value)

    def value(self, step) -> float:
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
