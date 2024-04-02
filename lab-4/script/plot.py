import csv  # reads .csv files
import matplotlib.pyplot as plt  # plots lists
import numpy as np  # numerical python library

# adc resolution
resolution = 0.0008


#
# Reads csv values with header row from file in filepath,
# and returns the header and datapoints in two seperate lists
#
def read_csv_file_with_header(filepath):
    header = []
    content = []

    with open(filepath) as csvfile:
        csvreader = csv.reader(csvfile)

        header = next(csvreader)

        for row in csvreader:
            #
            # Creates a list of the values on each row converted to floats
            #
            values = [float(value) for value in row]
            content.append(values)

        return header, content


def bode_plot(data, title="", save_plot=True, show_plot=True):
    frequency = [row[0] for row in data]  # frequency [Hz]
    # ch1 = [row[1] for row in data]        # filter input signal [V]
    ch2 = [row[2] for row in data]  # filter output signal [V]

    # plt.plot(frequency, ch1, 'y', linewidth=3, label=r"$X(f)$")
    plt.plot(frequency, ch2, linewidth=3, label=r"$Y(f)$")

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")

    plt.legend()
    plt.xscale("log")
    plt.grid(True)

    if not title == "":
        plt.title(title)
    else:
        title = "bode-plot"

    if save_plot:
        plt.savefig(
            "lab-4/img/" + str(title) + ".png",
            dpi=300,
            bbox_inches="tight",
        )

    if show_plot:
        plt.show()


def spectrum_plot(data, sample_period, title="", save_plot=True, show_plot=True):
    NFFT = len(data)
    Fs = 1 / sample_period
    df = Fs / NFFT

    f = np.arange(-Fs / 2, Fs / 2, df)  # frequency [Hz]
    f /= 1000  # frequency [kHz]

    # plt.figure().set_figwidth(12)
    plt.plot(f, data, linewidth=3, label=r"$S_x(f)$")

    plt.xlabel("Frekvens [kHz]")
    plt.ylabel("Relativ effekt [dB]")
    plt.title(title)

    plt.legend()
    plt.grid()

    if not title == "":
        plt.title(title)
    else:
        title = "spectrum-plot"

    if save_plot:
        plt.savefig(
            "lab-4/img/" + str(title) + ".png",
            dpi=300,
            bbox_inches="tight",
        )

    # allows modification of the plot outside of the function
    if show_plot:
        plt.show()


def time_plot(data, sample_period, title="", save_plot=True, show_plot=True):
    t = np.arange(0, (len(data) - 0.5) * sample_period, sample_period)  # time [s]
    t *= 1e3  # time [ms]

    # convert data to voltage [V]
    voltage = []
    for val in data:
        voltage.append(val * resolution)

    plt.plot(t, voltage, linewidth=3, marker=".", markersize=12, label=r"$x(t)$")

    plt.xlabel("Tid [ms]")
    plt.ylabel("Spenning [V]")

    plt.legend()
    plt.tight_layout()
    plt.grid()

    if not title == "":
        plt.title(title)
    else:
        title = "time-plot"

    if save_plot:
        plt.savefig(
            "lab-4/img/" + str(title) + ".png",
            dpi=300,
            bbox_inches="tight",
        )

    if show_plot:
        plt.show()


if __name__ == "__main__":
    _, data_A = read_csv_file_with_header("lab-4/data/bode-plot-filter-A.csv")
    bode_plot(data_A, "Bodeplot A", True, False)

    _, data_B = read_csv_file_with_header("lab-4/data/bode-plot-filter-B.csv")
    bode_plot(data_B, "Bodeplot B", True, False)

    spectrum_plot([1, 2, 1, 2, 1], 1, "", True, False)

    time_plot([1, 2, 1, 0, -1], 1, "", True, False)
