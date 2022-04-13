import glfw
import numpy as np


class Light:
    def __init__(self):
        self.color = [0, 0, 0]
        self.time = 0
        self.time_period = 5000

        self.factor = (1 / self.time_period) * 0.5


        self.day_color = [0.5, 0.6, 0.6]

        self.light_pos = [
                          [700, 700, 700],
                          [-700, -700, -700],
                          [-700, -700, -700],
                          [-700, -700, -700],
                        #   [10, 10, 10],
                        #   [10, 10, 10],
                        #   [10, 10, 10],
                          ]
        # self.light_pos = [[700, 700, 700],
        #                   [-60, 7, -30],
        #                   [15, 7, 55],
        #                   [10, 7, -10]]


        # https://math.hws.edu/graphicsbook/c7/s2.html
        self.light_atten = [[0, 0.00001, 0.0000004],
                            [0.4, 0.001, 0.0005],
                            [0.4, 0.001, 0.004],
                            [0.4, 0.001, 0.004]]


        self.num_light_src = 4


    def get_color(self):
        self.time = glfw.get_time() * 1000
        self.time %= self.time_period * 3

        # Dark
        if 0 <= self.time < 1*self.time_period:
            self.color = [0.5, 0.6, 0.6]

        # Light
        elif 1 * self.time_period <= self.time < self.time_period * 2.5:
            self.color = [0.2, 0.3, 0.3]

        return self.color
