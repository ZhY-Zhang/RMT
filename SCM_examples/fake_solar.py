import numpy as np
import matplotlib.pyplot as plt

# field
N = 5
DAYS = 365 * 25


def decline_func(values: np.ndarray, a: float, r: float) -> np.ndarray:
    # y = 1 / (1 + a * (exp(r * t) - 1))
    decline_factor = a * (np.exp(r * values) - 1)
    return 1 / (1 + decline_factor)


class SolarPanels:

    def __init__(self, size: int) -> None:
        self.size = size
        # properties
        self.age = np.zeros(size)              # days
        self.deposit = np.zeros(size)
        self.broken = np.zeros(size)
        self.obstacle = np.zeros(size)

    def solar_power(self, sunlight: float, rain: float, sandstorm: float) -> float:
        return sunlight * (1 - 0.2 * rain) * (1 - 0.4 * sandstorm)

    def electric_power(self, solar_power: float, deposit: np.ndarray, decay: np.ndarray) -> np.ndarray:
        return solar_power * deposit * decay

    def update(self, sunlight: float, rain: float, sandstorm: float) -> np.ndarray:
        """
        0 <= sunlight <= 1, 0 <= rain <= 1, 0 <= sandstorm <= 1
        """
        # update properties
        self.deposit += sandstorm * 6e-3
        '''
        new_broken = np.where(np.random.random(self.size) <= 1e-6, 1, 0)
        self.broken = np.logical_or(self.broken, new_broken)
        new_obstacle = np.where(np.random.random(self.size) <= 1e-4, 1, 0)
        self.obstacle = np.logical_or(self.obstacle, new_obstacle)
        '''
        # repair and clean
        if np.random.random() < 1e-2:
            self.deposit = np.zeros(self.size)
        '''
        if np.random.random() < 1e-2:
            self.broken = np.zeros(self.size)
        if np.random.random() < 1e-2:
            self.obstacle = np.zeros(self.size)
        '''
        # calculate comprehensive factors
        solar_power = self.solar_power(sunlight, rain, sandstorm)
        gain_deposit = decline_func(self.deposit, 0.1, 2)
        gain_decay = decline_func(self.age / 365, 0.01, 0.1) * (1 - self.broken)
        # return output power
        self.age += 1
        return self.electric_power(solar_power, gain_deposit, gain_decay)


if __name__ == '__main__':
    time_array = np.arange(DAYS)
    # sunlight
    sun_base = 0.85 + 0.1 * np.sin(2 * np.pi / 365 * time_array)
    sun_noise = 0.05 * np.random.normal(size=DAYS)
    sun_array = np.clip(sun_base + sun_noise, 0, 1)
    # rain
    rain_array = np.random.random(DAYS) * np.where(np.random.random(DAYS) < 130 / 360, 1, 0)
    # sandstorm
    sand_array = np.random.random(DAYS) * np.where(np.random.random(DAYS) < 80 / 360, 1, 0)

    output = np.empty((N, DAYS))
    sps = SolarPanels(N)
    for t in range(DAYS):
        output[:, t] = sps.update(sun_array[t], rain_array[t], sand_array[t])

    plt.plot(time_array, sun_array, linewidth=1)
    # plt.scatter(time_array, rain_array, s=1)
    # plt.scatter(time_array, sand_array, s=1)
    for y in output:
        plt.plot(time_array, y, linewidth=1)
        break
    plt.show()
