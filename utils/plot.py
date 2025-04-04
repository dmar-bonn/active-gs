import click
import seaborn as sb
import matplotlib.pyplot as plt
import pandas
import os
import numpy as np
import json
from matplotlib.ticker import FormatStrFormatter

PLOT_FONT_SIZE = 60
PLOT_LEGEND_SIZE = 60
PLOT_TICKS_SIZE = 60
PLOT_LINE_WIDTH = 10
IMG_FORMAT = "png"

plt.rcParams["font.weight"] = "bold"
plt.rcParams["figure.figsize"] = [18, 25]
plt.rcParams["figure.autolayout"] = False
sb.set_palette("bright")

planner_name = {
    "exploration": "FBE",
    "fisherrf": "FisherRF",
    "naruto": "NARUTO",
    "confidence": "Ours",
    "confidence_wo_roi": "Ours (w/o ROI)",
    "confidence_ablation": "Ours*",
}


@click.command()
@click.option("--data_folder", required=True, type=str, help="path to experiment")
def main(data_folder):
    total_df_time = pandas.DataFrame(
        {
            "Planner Type": [],
            "PSNR": [],
            "Completeness Ratio": [],
            "Mission Time": [],
        }
    )


    planner_list = [
        planner
        for planner in os.listdir(data_folder)
        if os.path.isdir(os.path.join(data_folder, planner))
    ]
    print(planner_list)

    data_dict = {}
    for planner in planner_list:
        planner_path = f"{data_folder}/{planner}"
        if os.path.exists(planner_path):
            id_list = [
                int(c)
                for c in os.listdir(planner_path)
                if os.path.isdir(os.path.join(planner_path, c))
            ]

            data_dict[planner] = {}

            for i, id in enumerate(id_list):
                result_data = f"{planner_path}/{id}/final_result.json"
                if os.path.exists(result_data):
                    with open(result_data, "r") as json_file:
                        result_data = json.load(json_file)

                    data_dict[planner][id] = result_data
    min_time, max_time = min_max_x_axis(data_dict, "time")

    for planner_type, planner_data in data_dict.items():
        for run_id, result_data in planner_data.items():
            # x axis
            mission_time = result_data["time"]

            # y axis
            psnr = result_data["mean_psnr"]
            completion_ratio = result_data["mesh_completion_ratio"]

            # interpolate via mission time
            mission_time_interp = np.arange(min_time, max_time, 10)
            psnr_time_interp = np.interp(
                mission_time_interp, mission_time, np.array(psnr)
            )
            completion_ratio_time_interp = np.interp(
                mission_time_interp,
                mission_time,
                np.array(completion_ratio),
            )
            for n in range(len(mission_time_interp)):
                dataframe_time = pandas.DataFrame(
                    {
                        "Planner Type": planner_name[planner_type],
                        "PSNR": psnr_time_interp[n],
                        "Completeness Ratio": completion_ratio_time_interp[n],
                        "Mission Time": mission_time_interp[n],
                    },
                    index=[i],
                )
                total_df_time = total_df_time._append(dataframe_time)


    fig, ax = plt.subplots()
    plot_ax(ax, "PSNR", total_df_time, x="Mission Time")
    plt.savefig(f"{data_folder}/psnr_time.{IMG_FORMAT}", bbox_inches="tight")
    plt.clf()


    fig, ax = plt.subplots()
    plot_ax(ax, "Completeness Ratio", total_df_time, x="Mission Time")
    plt.savefig(
        f"{data_folder}/completion_ratio_time.{IMG_FORMAT}", bbox_inches="tight"
    )
    plt.clf()


def plot_ax(ax, metric, dataframe, x):

    sb.lineplot(
        dataframe,
        x=x,
        y=metric,
        hue="Planner Type",
        style="Planner Type",
        linewidth=PLOT_LINE_WIDTH,
        ax=ax,
        errorbar=("sd", 1),
        palette=["C3", "C0", "C4", "C2", "C7", "C5"],
        dashes=[
            "",
            "",
            "",
            "",
            "",
            "",
        ],
    )
    ax.set_ylabel(metric, fontsize=PLOT_FONT_SIZE, weight="bold")
    ax.set_xlabel(x, fontsize=PLOT_FONT_SIZE, weight="bold")
    if metric == "PSNR":
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    ax.tick_params(axis="both", labelsize=PLOT_TICKS_SIZE)


def min_max_x_axis(data, label):
    min_x, max_x = float("inf"), float("-inf")
    for planner_type, planner_data in data.items():
        for run_id, result_data in planner_data.items():
            x_data = result_data[label]
            min_x = min(min_x, min(x_data))
            max_x = max(max_x, max(x_data))
    return min_x, max_x


if __name__ == "__main__":
    main()
