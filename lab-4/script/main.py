import sys
import os
from scipy.signal import hann
import numpy as np

from raspi_import import raspi_import
from plot import read_csv_with_header, bode_plot, spectrum_plot, time_plot
from fft import calc_spectrum


missing_file_path_error = (
    "\nMissing file path\n\nUsage example: python main.py foo.bin\n"
)

BIT_RESOLUTION = 4096
V_REF = 3.3
DC_OFFSET = 1


def find_radial_speed(sampling_period, data):

    dc_component = DC_OFFSET / V_REF * BIT_RESOLUTION
    data = data - dc_component

    time_plot(
        data, sample_period, title="Time plot of data", save_plot=True, show_plot=True
    )

    I = data[row][col]
    Q = data[row][col + 1]

    complex_data = I + 1j * Q

    lower_bounds = 0
    upper_bounds = -1
    data_slice = complex_data[lower_bounds:upper_bounds]

    NFFT = 524288

    data_spec, _ = calc_spectrum(data_slice, NFFT)
    spectrum_plot(data_spec, sample_period, "spectrum", save_plot=False, show_plot=True)

    data_windowed = data_slice * hann(len(data_slice))

    data_spec, _ = calc_spectrum(data_windowed, NFFT)
    spectrum_plot(
        data_spec,
        sample_period,
        "spectrum with Hanning window",
        save_plot=False,
        show_plot=True,
    )

    f0 = 24.13 * 10**9  # center frequency = 24,13 [GHz]
    c = 299792458  # speed of light [m/s]
    fD = np.argmax(data_windowed)  # Doppler frequency shift [Hz]
    v_r = fD * c / (2 * f0)  # radial speed [m/s]

    return v_r


if __name__ == "__main__":

    data_path = "C:\Users\tord_\OneDrive - NTNU\MTELSYS_2021-2026\S6_V2024\ttt4280-sensors-and-instrumentation\sensor-lab\lab-4\data"

    speed_folders = ["fast-towards", "slow-towards", "slow-away"]
    measurment_files = ["m1.bin", "m2.bin", "m3.bin", "m4.bin"]

    mean_speeds = []
    var_speeds = []

    for speed in speed_folders:
        radial_speeds = []

        for measurment in measurment_files:

            sampling_periode, data = raspi_import(os.path.join(data_path, speed, measurment))

            radial_speed = find_radial_speed(sampling_period, data)
            radial_speeds.append(radial_speed)

        my = np.mean(radial_speeds)
        mean_speeds.append(my)

        std = np.std(radial_speeds)
        var_speeds.append(std**2)

    print(mean_speeds)
    print(var_speeds)
