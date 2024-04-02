import sys
import os
from scipy.signal import hann, detrend
import numpy as np

from raspi_import import raspi_import
from plot import read_csv_file_with_header, bode_plot, spectrum_plot, time_plot
from fft import calc_spectrum


missing_file_path_error = (
    "\nMissing file path\n\nUsage example: python main.py foo.bin\n"
)


def find_radial_speed(sampling_period, data):

    I = data[5000:, 3]
    Q = data[5000:, 4]

    I = detrend(I)
    Q = detrend(Q)

    time_plot(
        I[10000:11000],
        sampling_period,
        title="Time plot of I",
        save_plot=False,
        show_plot=False,
    )
    time_plot(
        Q[10000:11000],
        sampling_period,
        title="Time plot",
        save_plot=False,
        show_plot=True,
    )

    I = I * hann(len(I))
    Q = Q * hann(len(Q))

    complex_data = I + 1j * Q

    NFFT = 524288

    data_spec, freqs = calc_spectrum(complex_data, NFFT)
    spectrum_plot(
        data_spec[int(NFFT / 2) - 15000 : int(NFFT / 2) + 15000],
        freqs[int(NFFT / 2) - 15000 : int(NFFT / 2) + 15000],
        sampling_period,
        "spectrum",
        save_plot=False,
        show_plot=True,
    )

    f0 = 24.13 * 10**9  # center frequency = 24,13 [GHz]
    c = 299792458  # speed of light [m/s]
    fD = np.argmax(data_spec)  # Doppler frequency shift [kHz]
    v_r = freqs[fD] * 1000 * c / (2 * f0)  # radial speed [m/s]

    return np.round(v_r*1000)/1000


if __name__ == "__main__":

    # sampling_period, data = raspi_import(
    #     "/Users/aasmundnorsett/Documents/NTNU/Semester6/Sensor/sensor-lab/lab-4/data/towards/3-1.bin"
    # )

    # radial_speed = find_radial_speed(sampling_period, data)

    # print("radial speed: " + str(radial_speed))

    data_path = "/Users/aasmundnorsett/Documents/NTNU/Semester6/Sensor/sensor-lab/lab-4/data"

    speed_folders = ["away-1", "away-2", "towards"]
    measurment_files = ["1.bin", "2.bin", "3.bin", "4.bin", "5.bin"]

    mean_speeds = []
    var_speeds = []

    for speed_folder in speed_folders:
        radial_speeds = []

        for measurment_file in measurment_files:

            sampling_period, data = raspi_import(os.path.join(data_path, speed_folder, measurment_file))

            radial_speed = find_radial_speed(sampling_period, data)
            radial_speeds.append(radial_speed)
            print(radial_speed)

        my = np.mean(radial_speeds)
        mean_speeds.append(my)

        std = np.std(radial_speeds)
        var_speeds.append(std**2)

    print(mean_speeds)
    print(var_speeds)
