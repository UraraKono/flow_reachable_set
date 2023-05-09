"""
Author: Victoria Edwards
Date: 10/06/2022

Purpose: This is to generate different waypoint configurations 
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

def unit_square():
    waypoints = [[-1.0, -1.0], [-1.0, -2.0], [1.0, -2.0 ], [1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0]]
    return(waypoints)

def lawnmower_pattern(bounds, direction = "short", num_legs = 5):
    """
    bounds = [x_min, x_max, y_min, y_max]
    direction: 
      - short: want the shorter side to be the main thrust
      - long: want the longer side to be the main thrust
    """
    dx = bounds[1] - bounds[0]
    dy = bounds[3] - bounds[2]

    waypoints = list()
    
    if direction == "short":
        x_or_y_len = np.argmin([np.fabs(dx), np.fabs(dy)])
    else:
        # Want long thrusts
        x_or_y_len = np.argmax([np.fabs(dx), np.fabs(dy)])

    if x_or_y_len != 0:
        dist = np.fabs(dy) / num_legs
        for i in range(num_legs):
            if i % 2 == 1: 
                waypoints.append([bounds[0], bounds[2] + dist * i]) # start of leg
                waypoints.append([bounds[1], bounds[2] + dist * i]) # end of leg
            else:
                waypoints.append([bounds[1], bounds[2] + dist * i]) # end of leg
                waypoints.append([bounds[0], bounds[2] + dist * i]) # start of leg

    else:
        dist = np.fabs(dx) / num_legs
        for i in range(num_legs):
            if i % 2 == 1: 
                waypoints.append([bounds[0] + dist * i, bounds[2]]) # start of leg
                waypoints.append([bounds[0] + dist * i, bounds[3]]) # end of leg
            else:
                waypoints.append([bounds[0] + dist * i, bounds[3]]) # end of leg
                waypoints.append([bounds[0] + dist * i, bounds[2]]) # start of leg
        
    return(waypoints)

def zamboni_pattern(bounds, direction = "short", num_legs = 5):
    """
    bounds = [x_min, x_max, y_min, y_max]
    direction: 
      - short: want the shorter side to be the main thrust
      - long: want the longer side to be the main thrust

    XXX: THIS ISN'T EXACTLY RIGHT BUT SHOULD WORK FOR OUR NEEDS RIGHT NOW
    """
    dx = bounds[1] - bounds[0]
    dy = bounds[3] - bounds[2]

    waypoints = list()
    
    if direction == "short":
        x_or_y_len = np.argmin([np.fabs(dx), np.fabs(dy)])
    else:
        # Want long thrusts
        x_or_y_len = np.argmax([np.fabs(dx), np.fabs(dy)])

    if x_or_y_len != 0:
        dist = np.fabs(dy) / (2.0 * num_legs)
        for i in range(num_legs):
            if i % 2 == 1: 
                waypoints.append([bounds[0], bounds[2] - (dist) * i]) # start of leg
                waypoints.append([bounds[1], bounds[2] - (dist) * i]) # end of leg
            else:
                waypoints.append([bounds[1], bounds[2] + (dist) * i]) # end of leg
                waypoints.append([bounds[0], bounds[2] + (dist) * i]) # start of leg

    else:
        dist = np.fabs(dx) / (2.0 * num_legs)
        for i in range(num_legs):
            if i % 2 == 1: 
                waypoints.append([bounds[0] - dist * i, bounds[2]]) # start of leg
                waypoints.append([bounds[0] - dist * i, bounds[3]]) # end of leg
            else:
                waypoints.append([bounds[0] + dist * i, bounds[3]]) # end of leg
                waypoints.append([bounds[0] + dist * i, bounds[2]]) # start of leg
        
    return(waypoints)


if __name__ == "__main__":
    print("SQUARE")
    waypoints = unit_square()
    print(unit_square())

    print("LAWNMOWER_PATTERN:")
    bounds = [-1.5, 1.8, -2.0, 0.0]
    waypoints = lawnmower_pattern(bounds, "short", num_legs = 10)
    print(waypoints)
    for i in range(len(waypoints)):
        plt.scatter(waypoints[i][0], waypoints[i][1], c = "k")

    for i in range(1, len(waypoints)):
        plt.plot([waypoints[i-1][0], waypoints[i][0]], [waypoints[i-1][1], waypoints[i][1]], color = "k")
    plt.show()

    print("ZAMBONI_PATTERN:")
    bounds = [-1.5, 2.5, -2.0, 0.0]
    waypoints = zamboni_pattern(bounds, "short", 10)
    print(waypoints)

    for i in range(len(waypoints)):
        plt.scatter(waypoints[i][0], waypoints[i][1])

    for i in range(1, len(waypoints)):
        plt.plot([waypoints[i-1][0], waypoints[i][0]], [waypoints[i-1][1], waypoints[i][1]])
        
    plt.show()
    