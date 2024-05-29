import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import re

cmap = 'gray'
print('Modules imported successfully...')

font = {'size': 14}
matplotlib.rc('font', **font)

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

class Segments:
    def __init__(self):
        self.flags = []
        self.segments = []
        #motor
        # h1 = [[2, 6], [0, 2]]  # 0
        # h2 = [[2, 6], [8, 10]]  # 1
        # h3 = [[2, 6], [16, 18]]  # 2
        # vl1 = [[1, 2], [3, 6]]  # 3 upper-left
        # vl2 = [[1, 2], [11, 14]]  # 4
        # vr1 = [[8, 9], [3, 6]]  # 5 upper-right
        # vr2 = [[8, 9], [11, 14]]  # 6

        #surface
        h1 = [[2, 6], [0, 1]]  # 0
        h2 = [[2, 6], [5, 6]]  # 1
        h3 = [[2, 6], [10, 11]]  # 2
        vl1 = [[0, 1], [1, 4]]  # 3 upper-left
        vl2 = [[0, 1], [6, 10]]  # 4
        vr1 = [[5, 6], [1, 4]]  # 5 upper-right
        vr2 = [[5, 6], [6, 10]]  # 6

        self.segments.append(h1)
        self.segments.append(h2)
        self.segments.append(h3)
        self.segments.append(vl1)
        self.segments.append(vl2)
        self.segments.append(vr1)
        self.segments.append(vr2)

    def digest(self, number):
        self.flags = []
        h, w = number.shape[:2]
        for a in range(len(self.segments)):
            seg = self.segments[a]
            xl, xh = seg[0]
            yl, yh = seg[1]
            sw = xh - xl
            sh = yh - yl
            # print(number[yl:yh, xl:xh])
            count = np.count_nonzero(number[yl:yh, xl:xh] < 38000)
            if count >= 2:  # Adjust active segment threshold to 4 pixels
                self.flags.append(a)
            # Debugging output
            # print(f"Segment {a} (xl={xl}, xh={xh}, yl={yl}, yh={yh}): count={count}, flags={self.flags}")

    def getNum(self):
        if self.flags == [0, 2, 3, 4, 5, 6]:
            return 0
        if self.flags == [5, 6]:
            return 1
        if self.flags == [0, 1, 2, 4, 5]:
            return 2
        if self.flags == [0, 1, 2, 5, 6]:
            return 3
        if self.flags == [1, 3, 5, 6]:
            return 4
        if self.flags == [0, 1, 2, 3, 6]:
            return 5
        if self.flags == [0, 1, 2, 3, 4, 6]:
            return 6
        if self.flags == [0, 5, 6]:
            return 7
        if self.flags == [0, 1, 2, 3, 4, 5, 6]:
            return 8
        if self.flags == [0, 1, 2, 3, 5, 6]:
            return 9
        return -1

def get_pressure_from_img_v2(path, verbose=True):
    img = Image.open(path)
    img_arr = np.array(img, dtype=np.int32)
    #motor
    # first_digit_img = img_arr[210:228, 261:271]
    # second_digit_img = img_arr[210:228, 272:282]
    # exponent_img = img_arr[210:228, 283:293]

    #surface
    first_digit_img = img_arr[233:244, 274:280]
    second_digit_img = img_arr[233:244, 281:287]
    exponent_img = img_arr[233:244, 288:294]
    s = Segments()
    s.digest(first_digit_img)
    first_digit = s.getNum()
    s.digest(second_digit_img)
    second_digit = s.getNum()
    s.digest(exponent_img)
    exponent = s.getNum()

    # if verbose:
    #     fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    #     ax[0].imshow(first_digit_img, cmap=cmap)
    #     ax[0].set_title('First Digit')
    #     ax[1].imshow(second_digit_img, cmap=cmap)
    #     ax[1].set_title('Second Digit')
    #     ax[2].imshow(exponent_img, cmap=cmap)
    #     ax[2].set_title('Decimal')
    #     plt.show()

    if verbose:
        if first_digit >= 0 and second_digit >= 0 and exponent >= 0:
            combined_value = float(f"{first_digit}{second_digit}.{exponent}")
        else:
            combined_value = "Invalid detection"
        print(combined_value)

    if first_digit < 0 or second_digit < 0 or exponent < 0:
        return np.nan
    return float(f"{first_digit}{second_digit}.{exponent}")

# Specify the directory containing the images
base_dir = Path(r"C:\Users\avazq\Downloads\P2_windmillTemp")
png_files = sorted(base_dir.glob("*.png"))

pressure_data = []

# Extract pressure values and time stamps
for png_path in png_files:
    pressure = get_pressure_from_img_v2(png_path, verbose=True)
    if not np.isnan(pressure):
        match = re.search(r'_(\d{3})\.png$', str(png_path))
        if match:
            time_value = int(match.group(1))
            pressure_data.append((time_value, pressure))

# Sort the data by time
pressure_data.sort()

# Extract time and pressure values for plotting
times, pressures = zip(*pressure_data)

# Plot the pressures over time
plt.figure(figsize=(10, 5))
plt.plot(times, pressures, marker='o', linestyle='-')
plt.xlabel('Time (seconds)')
plt.ylabel('Temperature (C)')
plt.title('Surface Temperature Over Time')
plt.grid(True)
plt.show()

# # Example usage for a single image
# base_dir = Path(r"C:\Users\avazq\Downloads\P2_windmillTemp")
# png_path = base_dir / "Scan017_P2_windmillTemp_027.png"
# get_pressure_from_img_v2(png_path, verbose=True)
