import os
import logging
import numpy as np
from json_tricks import load
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter, MaxNLocator
from mdutils import MdUtils
from rich.table import Table, Column
from rich.console import Console
from rich import box
import pandas as pd


class SimAnalyzer:

    def __init__(self,
                 simulations: list,
                 input_dir: str,
                 output_dir: str,
                 logger,
                 extension="pdf",
                 show_plot=False,
                 show_data=False):
        self.sims = []
        for sim_name, file in simulations:
            with open(input_dir + "/" + file, 'r') as fp:
                self.sims.append((sim_name, load(fp)))

        self.output_dir = os.path.join(output_dir, '')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.add_img_to_report = extension == ".png"
        self.show_plot = show_plot
        self.show_data = show_data
        self.logger = logger
        self.console = Console(record=True)
        self.output_list = []
        self.ext = "." + extension

        matplotlib_logger = logging.getLogger('matplotlib')
        matplotlib_logger.setLevel(logging.ERROR)

        self.output_report = MdUtils(file_name=output_dir + '/report', title='Simulation Report')

    def _add_img_to_report(self, title, img_filename, level=1):
        if self.add_img_to_report:
            self.output_report.new_header(level=level, title=title)
            self.output_report.new_paragraph("![](" + img_filename + ")")

    def _add_text_to_report(self, title, text, level=1):
        self.output_report.new_header(level=level, title=title)
        self.output_report.new_paragraph(text)

    def _save_show_plot(self, output_filename):
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_filename))
        if self.show_plot:
            plt.show()
        plt.close()

    def _plot_acc_loss(self, phase, title, ylabel, legend_loc, key):
        fig, ax = plt.subplots()
        for name, sim in self.sims:
            ys = []
            for status in sim["status"]:
                ys.append(status["var"][phase]["model_metrics"][key])
            y_mean = [np.mean(y) for y in zip(*ys)]
            y_std = [np.std(y) for y in zip(*ys)]
            x = range(1, len(y_mean) + 1)
            ax.errorbar(x, y_mean, fmt='-o', yerr=y_std, label=name)

            if self.show_data:
                self.logger.info("{} - agg " + title + ": {}".format(name, ys))

        ax.set(title=title, xlabel="round", ylabel=ylabel)
        ax.legend(loc=legend_loc)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True)

    def plot_metric(self, phase="eval"):
        title = f"Metric ({phase})"
        self._plot_acc_loss(phase=phase, title=title,
                            ylabel="metric %", legend_loc=4, key="agg_metric")

        output_filename = "agg_metric_" + phase + self.ext
        self._save_show_plot(output_filename)
        self._add_img_to_report(title, output_filename)

    def plot_loss(self, phase="eval"):
        title = f"Loss ({phase})"
        self._plot_acc_loss(phase=phase, title=title,
                            ylabel="loss", legend_loc=1, key="agg_loss")

        output_filename = "agg_loss_" + phase + self.ext
        self._save_show_plot(output_filename)
        self._add_img_to_report(title, output_filename)

    def _plot_round_times(self, phase, title, keys, ylabel="time [s]", legend_loc=1):
        fig, ax = plt.subplots()
        for i, (name, sim) in enumerate(self.sims):
            ys = []
            for status in sim["status"]:
                val = 0
                for key in keys:
                    val += status["var"][phase]["times"][key]
                ys.append(np.amax(val, axis=1))
            y_mean = [np.mean(y) for y in zip(*ys)]
            y_std = [np.std(y) for y in zip(*ys)]
            x = range(1, len(y_mean) + 1)
            ax.errorbar(x, y_mean, fmt='-o', yerr=y_std, label=name)
        ax.set(title=title, xlabel='round', ylabel=ylabel)
        ax.legend(loc=legend_loc)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True)

    def plot_computation_time(self, phase="fit"):
        title = f"Computation Time ({phase})"
        self._plot_round_times(phase=phase, title=title, keys=["computation"])

        output_filename = "rt_computation_" + phase + self.ext
        self._save_show_plot(output_filename)
        self._add_img_to_report(title, output_filename)

    def plot_communication_time(self, phase="fit"):
        title = f"Communication Time ({phase})"
        self._plot_round_times(phase=phase, title=title, keys=["communication_upload", "communication_distribution"])

        output_filename = "rt_communication_" + phase + self.ext
        self._save_show_plot(output_filename)
        self._add_img_to_report(title, output_filename)

    def plot_total_time(self, phase="fit"):
        title = f"Total Time ({phase})"
        self._plot_round_times(phase=phase, title=title, keys=["computation", "communication_upload", "communication_distribution"])

        output_filename = "rt_total_" + phase + self.ext
        self._save_show_plot(output_filename)
        self._add_img_to_report(title, output_filename)

    def _plot_round_consumptions(self, phase, title, keys, ylabel, legend_loc=1):
        fig, ax = plt.subplots()
        for name, sim in self.sims:
            ys = []
            for status in sim["status"]:
                val = 0
                for key in keys:
                    val += status["var"][phase]["consumption"][key]
                ys.append(np.sum(val, axis=1))
            y_mean = [np.mean(y) for y in zip(*ys)]
            y_std = [np.std(y) for y in zip(*ys)]
            x = range(1, len(y_mean) + 1)
            ax.errorbar(x, y_mean, fmt='-o', yerr=y_std, label=name)

            if self.show_data:
                self.logger.info("{} - " + title + ": \n{}\nagg: {}".format(name, val, y_mean))
        ax.set(title=title, xlabel='round', ylabel=ylabel)
        ax.legend(loc=legend_loc)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True)

    def plot_resources_consumption(self, phase="fit"):
        title = f"Resources Consumption ({phase})"
        self._plot_round_consumptions(phase=phase, title=title, keys=["resources"], ylabel="iters")

        output_filename = "consumption_resources_" + phase + self.ext
        self._save_show_plot(output_filename)
        self._add_img_to_report(title, output_filename)

    def plot_network_consumption(self, phase="fit"):
        title = f"Network Consumption ({phase})"
        self._plot_round_consumptions(phase=phase, title=title, keys=["network_upload", "network_distribution"], ylabel="params")

        output_filename = "consumption_network_" + phase + self.ext
        self._save_show_plot(output_filename)
        self._add_img_to_report(title, output_filename)

    def plot_energy_consumption(self, phase="fit"):
        title = f"Energy Consumption ({phase})"
        self._plot_round_consumptions(phase=phase, title=title, keys=["energy"], ylabel="mA")

        output_filename = "consumption_energy_" + phase + self.ext
        self._save_show_plot(output_filename)
        self._add_img_to_report(title, output_filename)

    def plot_available_devices(self):
        title = "Available Devices"
        ylabel = "devices"
        legend_loc = 1

        fig, ax = plt.subplots()
        for name, sim in self.sims:
            ys = []
            for status in sim["status"]:
                val = status["con"]["devs"]["availability"]
                ys.append(np.sum(val, axis=1))
            y_mean = [np.mean(y) for y in zip(*ys)]
            y_std = [np.std(y) for y in zip(*ys)]
            x = range(1, len(y_mean) + 1)
            ax.errorbar(x, y_mean, fmt='-o', yerr=y_std, label=name)
        ax.set(title=title, xlabel='round', ylabel=ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc=legend_loc)
        ax.grid(True)

        output_filename = "devices_available" + self.ext
        self._save_show_plot(output_filename)
        self._add_img_to_report(title, output_filename)

    def plot_selected_devices(self, phase="fit"):
        title = f"Selected Devices ({phase})"
        ylabel = "devices"
        legend_loc = 1

        fig, ax = plt.subplots()
        for name, sim in self.sims:
            ys = []
            for status in sim["status"]:
                val = status["var"][phase]["devs"]["selected"]
                ys.append(np.sum(val, axis=1))
            y_mean = [np.mean(y) for y in zip(*ys)]
            y_std = [np.std(y) for y in zip(*ys)]
            x = range(1, len(y_mean) + 1)
            ax.errorbar(x, y_mean, fmt='-o', yerr=y_std, label=name)
        ax.set(title=title, xlabel='round', ylabel=ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc=legend_loc)
        ax.grid(True)

        output_filename = "devices_selected_" + phase + self.ext
        self._save_show_plot(output_filename)
        self._add_img_to_report(title, output_filename)

    def plot_available_failed_devices(self):
        title = "Available and Failed Devices"
        ylabel = "devices"
        legend_loc = 1

        fig, ax = plt.subplots()
        for name, sim in self.sims:
            ys = []
            for status in sim["status"]:
                val = status["con"]["devs"]["availability"] & status["con"]["devs"]["failures"]
                ys.append(np.sum(val, axis=1))
            y_mean = [np.mean(y) for y in zip(*ys)]
            y_std = [np.std(y) for y in zip(*ys)]
            x = range(1, len(y_mean) + 1)
            ax.errorbar(x, y_mean, fmt='-o', yerr=y_std, label=name)
        ax.set(title=title, xlabel='round', ylabel=ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc=legend_loc)
        ax.grid(True)

        output_filename = "devices_available_failed" + self.ext
        self._save_show_plot(output_filename)
        self._add_img_to_report(title, output_filename)

    def plot_selected_successful_devices(self, phase="fit"):
        title = f"Selected and Successful Devices ({phase})"
        ylabel = "devices"
        legend_loc = 1

        fig, ax = plt.subplots()
        for name, sim in self.sims:
            ys = []
            for status in sim["status"]:
                val = status["var"][phase]["devs"]["selected"] - \
                      (status["var"][phase]["devs"]["selected"] & status["con"]["devs"]["failures"])
                ys.append(np.sum(val, axis=1))
                print(np.sum(val, axis=0))
            y_mean = [np.mean(y) for y in zip(*ys)]
            y_std = [np.std(y) for y in zip(*ys)]
            x = range(1, len(y_mean) + 1)
            ax.errorbar(x, y_mean, fmt='-o', yerr=y_std, label=name)
        ax.set(title=title, xlabel='round', ylabel=ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc=legend_loc)
        ax.grid(True)

        output_filename = "devices_selected_successful_" + phase + self.ext
        self._save_show_plot(output_filename)
        self._add_img_to_report(title, output_filename)

    def _plot_round_configs(self, phase, title, key, ylabel, legend_loc=1):
        self.output_report.new_header(level=1, title=title)
        for name, sim in self.sims:
            for i, status in enumerate(sim["status"]):
                fig, ax = plt.subplots()
                val1 = status["var"][phase]["upd_opt_configs"]["global"][key]
                val2 = status["var"][phase]["upd_opt_configs"]["local"][key]
                y1 = np.sum(val1, axis=1)
                y2 = np.sum(val2, axis=1)
                x = range(1, y1.shape[0] + 1)
                ax.plot(x, y1, '--o', label="$" + name + "_{global}$")
                ax.plot(x, y2, '-o', label="$" + name + "_{local}$")
                ax.set(title=title, xlabel='round', ylabel=ylabel)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.legend(loc=legend_loc)
                ax.grid()

                output_filename = "config_" + key + "_" + name + "_" + str(i) + "_" + phase + self.ext
                self._save_show_plot(output_filename)
                self._add_img_to_report(str(i), output_filename, level=2)

    def plot_epochs_config(self, phase="fit"):
        title = f"Epochs ({phase})"
        self._plot_round_configs(phase=phase, title=title, key="epochs", ylabel="epochs")

    def plot_batch_size_config(self, phase="fit"):
        title = f"Batch Size ({phase})"
        self._plot_round_configs(phase=phase, title=title, key="batch_size", ylabel="size")

    def plot_num_examples_config(self, phase="fit"):
        title = f"Num Examples ({phase})"
        self._plot_round_configs(phase=phase, title=title, key="num_examples", ylabel="examples")

    def plot_matrix_devices(self, phase="fit"):
        title = f"Devices Matrix ({phase})"
        self.output_report.new_header(level=1, title=title)

        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        cm = LinearSegmentedColormap.from_list("dev_cmap", colors)
        values = [0, 1, 2]
        labels = ["Not available", "Available", "Selected"]

        for name, sim in self.sims:
            for j, status in enumerate(sim["status"]):
                fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False)
                i = 0
                ax[i][0].set(title=title + " " + name, xlabel='devices', ylabel='round')
                im = ax[i][0].imshow(
                    status["con"]["devs"]["availability"].T + status["var"][phase]["devs"]["selected"].T,
                    interpolation=None, cmap=cm)
                i += 1
                colors = [im.cmap(im.norm(value)) for value in values]
                patches = [mpatches.Patch(color=colors[i], label=labels[i].format(l=values[i]))
                           for i in range(len(values))]
                plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

                output_filename = "matrix_devices_" + name + "_" + str(j) + "_" + phase + self.ext
                self._save_show_plot(output_filename)
                self._add_img_to_report(str(j), output_filename, level=2)

    def plot_devices_bar_availability(self, phase="fit"):
        title = f"Availability ({phase})"
        xlabel = "device"
        ylabel = "density"
        legend_loc = 4

        self.output_report.new_header(level=1, title=title)
        for name, sim in self.sims:
            for i, status in enumerate(sim["status"]):
                fig, ax = plt.subplots()

                val = status["con"]["devs"]["availability"]
                y = np.sum(val, axis=0) / val.shape[0]
                x = range(0, y.shape[0])
                ax.bar(x, y, label=name, alpha=1)
                ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.legend(loc=legend_loc)
                ax.grid()

                output_filename = "devs_bar_availability_" + name + "_" + str(i) + "_" + phase + self.ext
                self._save_show_plot(output_filename)
                self._add_img_to_report(str(i), output_filename, level=2)

    def plot_devices_bar_failures(self, phase="fit"):
        title = f"Failures ({phase})"
        xlabel = "device"
        ylabel = "density"
        legend_loc = 4

        self.output_report.new_header(level=1, title=title)
        for name, sim in self.sims:
            for i, status in enumerate(sim["status"]):
                fig, ax = plt.subplots()

                val = status["con"]["devs"]["failures"]
                y = np.sum(val, axis=0) / val.shape[0]
                x = range(0, y.shape[0])
                ax.bar(x, y, label=name, alpha=1)
                ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.legend(loc=legend_loc)
                ax.grid()

                output_filename = "devs_bar_failures_" + name + "_" + str(i) + "_" + phase + self.ext
                self._save_show_plot(output_filename)
                self._add_img_to_report(str(i), output_filename, level=2)

    def plot_devices_bar_selected(self, phase="fit"):
        title = f"Selected ({phase})"
        self.output_report.new_header(level=1, title=title)

        for name, sim in self.sims:
            for i, status in enumerate(sim["status"]):
                fig, ax = plt.subplots()

                # selected
                val = status["var"][phase]["devs"]["selected"]
                y = np.sum(val, axis=0) / val.shape[0]
                x = range(0, y.shape[0])
                ax.bar(x, y, label=name, alpha=1)
                ax.set(title=title, xlabel="device", ylabel="density")
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.legend(loc=4)
                ax.grid()

                output_filename = "devs_bar_selected_" + name + "_" + str(i) + "_" + phase + self.ext
                self._save_show_plot(output_filename)
                self._add_img_to_report(str(i), output_filename, level=2)

    def plot_devices_data_size(self):
        title = "Local Data Size"
        self.output_report.new_header(level=1, title=title)

        for name, sim in self.sims:
            for i, status in enumerate(sim["status"]):
                fig, ax = plt.subplots()

                # local data
                val = status["con"]["devs"]["local_data_sizes"]
                ax.hist(val, label=name, density=False, alpha=1)
                ax.set(title=title, xlabel='# examples', ylabel='density')
                ax.legend(loc=1)
                ax.grid(True)

                output_filename = "devs_data_size_" + name + "_" + str(i) + self.ext
                self._save_show_plot(output_filename)
                self._add_img_to_report(str(i), output_filename, level=2)

    def plot_devices_ips(self):
        title = "IPS"
        self.output_report.new_header(level=1, title=title)

        for name, sim in self.sims:
            for i, status in enumerate(sim["status"]):
                fig, ax = plt.subplots()

                # IPS
                val = status["con"]["devs"]["ips"]
                ax.hist(val, label=name, density=False, alpha=1)
                ax.set(title=title, xlabel='IPS', ylabel='density')
                ax.legend(loc=1)
                ax.grid(True)

                output_filename = "devs_ips_" + name + "_" + str(i) + self.ext
                self._save_show_plot(output_filename)
                self._add_img_to_report(str(i), output_filename, level=2)

    def plot_devices_available_energy(self):
        title = "Energy"
        self.output_report.new_header(level=1, title=title)

        for name, sim in self.sims:
            for i, status in enumerate(sim["status"]):
                fig, ax = plt.subplots()

                # energy
                val = np.mean(status["con"]["devs"]["energy"], axis=1)
                ax.hist(val, label=name, density=False, alpha=1)
                ax.set(title=title, xlabel='mAh', ylabel='density')
                ax.legend(loc=1)
                ax.grid(True)

                output_filename = "devs_available_energy_" + name + "_" + str(i) + self.ext
                self._save_show_plot(output_filename)
                self._add_img_to_report(str(i), output_filename, level=2)

    def plot_devices_network_speed(self):
        title = "Network Speed"
        self.output_report.new_header(level=1, title=title)

        for name, sim in self.sims:
            for i, status in enumerate(sim["status"]):
                fig, ax = plt.subplots()

                # network
                val = np.mean(status["con"]["devs"]["net_speed"], axis=1)
                ax.hist(val, label=name, density=False, alpha=1)
                ax.set(title=title, xlabel='params/s', ylabel='density')
                ax.legend(loc=1)
                ax.grid(True)

                output_filename = "devs_network_speed_" + name + "_" + str(i) + self.ext
                self._save_show_plot(output_filename)
                self._add_img_to_report(str(i), output_filename, level=2)

    def plot_devices_data_distribution(self):
        title = "Data Distribution"
        self.output_report.new_header(level=1, title=title)
        num_classes = []
        for name, sim in self.sims:
            for i, status in enumerate(sim["status"]):
                fig, ax = plt.subplots(1)

                # local data
                local_data_stats = status["con"]["devs"]["local_data_stats"]
                for x in local_data_stats:
                    num_classes.append(len(set(x)))
                for j, local_data in enumerate(local_data_stats):
                    cumul = 0
                    for count in local_data:
                        ax.barh(j, count, left=cumul)
                        cumul += count

                ax.grid()
                ax.set(title=title, xlabel='#', ylabel='device')

                output_filename = "devs_local_data_" + name + "_" + str(i) + self.ext
                self._save_show_plot(output_filename)
                self._add_img_to_report(str(i), output_filename, level=2)

    def _init_console_table(self, title, min_max=True,
                            column_names=("mean", "std", "min", "max"),
                            column_styles=("magenta", "cyan", "blue", "bright_magenta", "bright_cyan")):

        columns = [Column(header="name", style="green"), Column(header="rep", style="italic yellow")]
        for i, (name, style) in enumerate(zip(column_names, column_styles)):
            if i >= 2 and not min_max:
                break
            columns.append(Column(header=name, style=style))
        return Table(*columns, title=title, box=box.SQUARE)

    def print_availability(self):
        table = self._init_console_table(title="AVAILABILITY", min_max=False)
        for name, sim in self.sims:
            means_avail = []
            stds_avail = []
            for i, status in enumerate(sim["status"]):
                availability = status["con"]["devs"]["availability"]
                mean_avail = availability.mean()
                std_avail = availability.std()
                means_avail.append(mean_avail)
                stds_avail.append(std_avail)
                table.add_row(name, str(i + 1), f"{mean_avail * 100:.2f}%", f"{std_avail:.2f}")
            table.add_row(f"[red]{name}[/]", "all", f"{np.mean(means_avail) * 100:.2f}%",
                          f"{np.mean(stds_avail):.2f}")
            self.output_list.append({"type": "availability", "name": name, "rep": "all",
                                     "mean": np.mean(means_avail), "std": np.mean(stds_avail)})
        self.console.print(table)

    def print_failures(self):
        table = self._init_console_table(title="FAILURES", min_max=False)
        for name, sim in self.sims:
            means_failures = []
            stds_failures = []
            for i, status in enumerate(sim["status"]):
                failures = status["con"]["devs"]["failures"]
                mean_failures = failures.mean()
                std_failures = failures.std()
                means_failures.append(mean_failures)
                stds_failures.append(std_failures)
                table.add_row(name, str(i + 1), f"{mean_failures * 100:.2f}%", f"{std_failures:.2f}")
            table.add_row(f"[red]{name}[/]", "all", f"{np.mean(means_failures) * 100:.2f}%",
                          f"{np.mean(stds_failures):.2f}")
            self.output_list.append({"type": "failures", "name": name, "rep": "all",
                                     "mean": np.mean(means_failures), "std": np.mean(stds_failures)})
        self.console.print(table)

    def print_ips(self):
        table = self._init_console_table(title="IPS \[iters/s]")
        for name, sim in self.sims:
            means_ips = []
            stds_ips = []
            for i, status in enumerate(sim["status"]):
                ips = status["con"]["devs"]["ips"]
                mean_ips = ips.mean()
                std_ips = ips.std()
                means_ips.append(mean_ips)
                stds_ips.append(std_ips)
                table.add_row(name, str(i + 1), f"{mean_ips:.2f}", f"{std_ips:.2f}",
                              f"{ips.min():.2f}", f"{ips.max():.2f}")
            table.add_row(f"[red]{name}[/]", "all", f"{np.mean(means_ips):.2f}", f"{np.mean(stds_ips):.2f}",
                          f"{np.min(means_ips):.2f}", f"{np.max(means_ips):.2f}")
            self.output_list.append({"type": "ips", "name": name, "rep": "all", "mean": np.mean(means_ips),
                                     "std": np.mean(stds_ips), "min": np.min(means_ips),
                                     "max": np.max(means_ips)})
        self.console.print(table)

    def print_energy(self):
        table = self._init_console_table(title="ENERGY \[mAh]")
        for name, sim in self.sims:
            means_energy = []
            stds_energy = []
            for i, status in enumerate(sim["status"]):
                energy = status["con"]["devs"]["energy"]
                mean_energy = energy.mean()
                std_energy = energy.std()
                means_energy.append(mean_energy)
                stds_energy.append(std_energy)
                table.add_row(name, str(i + 1), f"{mean_energy:.2f}", f"{std_energy:.2f}",
                              f"{energy.min():.2f}", f"{energy.max():.2f}")
            table.add_row(f"[red]{name}[/]", "all", f"{np.mean(means_energy):.2f}",
                          f"{np.mean(stds_energy):.2f}", f"{np.min(means_energy):.2f}", f"{np.max(means_energy):.2f}")
            self.output_list.append({"type": "energy", "name": name, "rep": "all", "mean": np.mean(means_energy),
                                     "std": np.mean(stds_energy), "min": np.min(means_energy),
                                     "max": np.max(means_energy)})
        self.console.print(table)

    def print_net_speed(self):
        table = self._init_console_table(title="NET SPEED \[params/s]")
        for name, sim in self.sims:
            means_net_speed = []
            stds_net_speed = []
            for i, status in enumerate(sim["status"]):
                net_speed = status["con"]["devs"]["net_speed"]
                mean_net_speed = net_speed.mean()
                std_net_speed = net_speed.std()
                means_net_speed.append(mean_net_speed)
                stds_net_speed.append(std_net_speed)
                table.add_row(name, str(i + 1), f"{mean_net_speed:.2f}", f"{std_net_speed:.2f}",
                              f"{net_speed.min():.2f}", f"{net_speed.max():.2f}")
            table.add_row(f"[red]{name}[/]", "all", f"{np.mean(means_net_speed):.2f}",
                          f"{np.mean(stds_net_speed):.2f}", f"{np.min(means_net_speed):.2f}",
                          f"{np.max(means_net_speed):.2f}")
            self.output_list.append({"type": "net_speed", "name": name, "rep": "all", "mean": np.mean(means_net_speed),
                                     "std": np.mean(stds_net_speed), "min": np.min(means_net_speed),
                                     "max": np.max(means_net_speed)})
        self.console.print(table)

    def print_local_data_size(self):
        table = self._init_console_table(title="LOCAL DATA SIZE \[samples]")
        for name, sim in self.sims:
            means_local_data_size = []
            stds_local_data_size = []
            for i, status in enumerate(sim["status"]):
                local_data_size = status["con"]["devs"]["local_data_sizes"]
                mean_local_data_size = local_data_size.mean()
                std_local_data_size = local_data_size.std()
                means_local_data_size.append(mean_local_data_size)
                stds_local_data_size.append(std_local_data_size)
                table.add_row(name, str(i + 1), f"{mean_local_data_size:.2f}", f"{std_local_data_size:.2f}",
                              f"{local_data_size.min():.2f}", f"{local_data_size.max():.2f}")
            table.add_row(f"[red]{name}[/]", "all", f"{np.mean(means_local_data_size):.2f}",
                          f"{np.mean(stds_local_data_size):.2f}", f"{np.min(means_local_data_size):.2f}",
                          f"{np.max(means_local_data_size):.2f}")
            self.output_list.append({"type": "local_data_size", "name": name, "rep": "all",
                                     "mean": np.mean(means_local_data_size), "std": np.mean(stds_local_data_size),
                                     "min": np.min(means_local_data_size), "max": np.max(means_local_data_size)})
        self.console.print(table)

    def print_model_params(self):
        table = self._init_console_table(column_names=["params"], title="MODEL PARAMS")
        for name, sim in self.sims:
            for i, status in enumerate(sim["status"]):
                table.add_row(name, str(i + 1), str(status["con"]["model"]["tot_weights"]))
        self.console.print(table)

    def print_selection(self, phase="fit"):
        table = self._init_console_table(column_names=["mean", "mean succ. devs\n(among selected ones)"],
                                         title=f"SELECTION ({phase})")
        for name, sim in self.sims:
            means_selected = []
            successful_devs_ratios = []
            for i, status in enumerate(sim["status"]):
                selected = status["var"][phase]["devs"]["selected"]
                successful_devs = status["var"][phase]["devs"]["selected"] - \
                                  (status["var"][phase]["devs"]["selected"] & status["con"]["devs"]["failures"])
                successful_devs_ratio = successful_devs.sum() / status["var"][phase]["devs"]["selected"].sum()
                means_selected.append(selected.mean())
                successful_devs_ratios.append(successful_devs_ratio)
                table.add_row(name, str(i + 1), f"{selected.mean() * 100:.2f}%", f"{successful_devs_ratio * 100:.2f}%")
            table.add_row(f"[red]{name}[/]", "all", f"{np.mean(means_selected) * 100:.2f}%",
                          f"{np.mean(successful_devs_ratios) * 100:.2f}%")
            self.output_list.append({"type": "selection", "phase": phase, "name": name, "rep": "all",
                                     "mean": np.mean(means_selected),
                                     "mean succ": np.mean(successful_devs_ratios)})
        self.console.print(table)

    def print_total_time(self, phase="fit"):
        table = self._init_console_table(column_names=["total", "mean round", "std round", "min", "max"],
                                         title=f"TOTAL TIME \[s] ({phase})")
        for name, sim in self.sims:
            tot_tts = []
            mean_rts = []
            std_rts = []
            for i, status in enumerate(sim["status"]):
                rt = status["var"][phase]["times"]["computation"] + status["var"][phase]["times"]["communication_upload"] + status["var"][phase]["times"]["communication_distribution"]
                max_round = np.amax(rt, axis=1)
                tot_tt = np.sum(max_round)
                mean_rt = np.mean(max_round)
                std_rt = np.std(max_round)
                min_rt = np.min(max_round)
                max_rt = np.max(max_round)
                tot_tts.append(tot_tt)
                mean_rts.append(mean_rt)
                std_rts.append(std_rt)
                table.add_row(name, str(i + 1), f"{tot_tt:.2f}", f"{mean_rt:.2f}", f"{std_rt:.2f}",
                              f"{min_rt:.2f}", f"{max_rt:.2f}")
            table.add_row(f"[red]{name}[/]", "all", f"{np.mean(tot_tts):.2f}", f"{np.mean(mean_rts):.2f}",
                          f"{np.mean(std_rts):.2f}", f"{np.min(tot_tts):.2f}", f"{np.max(tot_tts):.2f}")
            self.output_list.append({"type": "times", "phase": phase, "name": name, "rep": "all",
                                     "mean": np.mean(tot_tts), "mean_round": np.mean(mean_rts),
                                     "std_round": np.mean(std_rts), "min": np.min(tot_tts), "max": np.max(tot_tts)})
        self.console.print(table)

    def print_resources_consumption(self, phase="fit"):
        table = self._init_console_table(column_names=["total", "mean round", "std round"],
                                         title=f"RESOURCES CONSUMPTION \[iters] ({phase})")
        for name, sim in self.sims:
            tot_rcs = []
            mean_rcs = []
            std_rcs = []
            for i, status in enumerate(sim["status"]):
                rc = status["var"][phase]["consumption"]["resources"]
                tot_rc = np.sum(rc)
                mean_rc = np.mean(np.sum(rc, axis=1))
                std_rc = np.std(np.sum(rc, axis=1))
                tot_rcs.append(tot_rc)
                mean_rcs.append(mean_rc)
                std_rcs.append(std_rc)
                table.add_row(name, str(i + 1), f"{tot_rc:.2f}", f"{mean_rc:.2f}", f"{std_rc:.2f}")
            table.add_row(f"[red]{name}[/]", "all", f"{np.mean(tot_rcs):.2f}", f"{np.mean(mean_rcs):.2f}",
                          f"{np.mean(std_rcs):.2f}")
            self.output_list.append({"type": "resources_consumption", "phase": phase, "name": name, "rep": "all",
                                     "mean": np.mean(tot_rcs), "mean_round": np.mean(mean_rcs),
                                     "std_round ": np.mean(std_rcs)})
        self.console.print(table)

    def print_energy_consumption(self, phase="fit"):
        table = self._init_console_table(column_names=["total", "mean round", "std round"],
                                         title=f"ENERGY CONSUMPTION \[mAh] ({phase})")
        for name, sim in self.sims:
            tot_ecs = []
            mean_ecs = []
            std_ecs = []
            for i, status in enumerate(sim["status"]):
                ec = status["var"][phase]["consumption"]["energy"]
                tot_ec = np.sum(ec)
                mean_ec = np.mean(np.sum(ec, axis=1))
                std_ec = np.std(np.sum(ec, axis=1))
                tot_ecs.append(tot_ec)
                mean_ecs.append(mean_ec)
                std_ecs.append(std_ec)
                table.add_row(name, str(i + 1), f"{tot_ec:.2e}", f"{mean_ec:.2e}", f"{std_ec:.2e}")
            table.add_row(f"[red]{name}[/]", "all", f"{np.mean(tot_ecs):.2e}", f"{np.mean(mean_ecs):.2e}",
                          f"{np.mean(std_ecs):.2e}")
            self.output_list.append({"type": "energy_consumption", "phase": phase, "name": name, "rep": "all",
                                     "mean": np.mean(tot_ecs), "mean_round": np.mean(mean_ecs),
                                     "std_round ": np.mean(std_ecs)})
        self.console.print(table)

    def print_network_consumption(self, phase="fit"):
        table = self._init_console_table(column_names=["total", "mean round", "std round"],
                                         title=f"NETWORK CONSUMPTION \[params] ({phase})")
        for name, sim in self.sims:
            tot_ncs = []
            mean_ncs = []
            std_ncs = []
            for i, status in enumerate(sim["status"]):
                nc = status["var"][phase]["consumption"]["network_upload"] + status["var"][phase]["consumption"]["network_distribution"]
                tot_nc = np.sum(nc)
                mean_nc = np.mean(np.sum(nc, axis=1))
                std_nc = np.std(np.sum(nc, axis=1))
                tot_ncs.append(tot_nc)
                mean_ncs.append(mean_nc)
                std_ncs.append(std_nc)
                table.add_row(name, str(i + 1), f"{tot_nc:.2e}", f"{mean_nc:.2e}", f"{std_nc:.2e}")
            table.add_row(f"[red]{name}[/]", "all", f"{np.mean(tot_ncs):.2e}", f"{np.mean(mean_ncs):.2e}",
                          f"{np.mean(std_ncs):.2e}")
            self.output_list.append({"type": "network_consumption", "phase": phase, "name": name, "rep": "all",
                                     "mean": np.mean(tot_ncs), "mean_round": np.mean(mean_ncs),
                                     "std_round": np.mean(std_ncs)})
        self.console.print(table)

    def print_metric(self, phase="eval", round=-1):

        if round == -1:
            round_column = "latest"
        else:
            round_column = "ROUND " + str(round)
        table = self._init_console_table(column_names=[round_column, "num rounds"], title=f"ACCURACY [%] ({phase})")
        for name, sim in self.sims:
            latest_accs = []
            rounds = []
            for i, status in enumerate(sim["status"]):
                latest_acc = status["var"][phase]["model_metrics"]["agg_metric"][round]
                round_acc = status["var"][phase]["model_metrics"]["agg_metric"].shape[0]
                latest_accs.append(latest_acc)
                rounds.append(round_acc)
                table.add_row(name, str(i + 1), f"{latest_acc:.2f}", f"{round_acc}")
            table.add_row(f"[red]{name}[/]", "all", f"{np.mean(latest_accs):.2f}", f"{np.mean(rounds):.2f}")
            self.output_list.append({"type": "accuracy", "phase": phase, "name": name, "rep": "all",
                                     "mean": np.mean(latest_accs), "num_rounds": np.mean(rounds)})
        self.console.print(table)

    def print_loss(self, phase="eval", round=-1):
        if round == -1:
            round_column = "latest"
        else:
            round_column = "ROUND " + str(round)
        table = self._init_console_table(column_names=[round_column, "num rounds"], title=f"LOSS ({phase})")
        for name, sim in self.sims:
            latest_losses = []
            rounds = []
            for i, status in enumerate(sim["status"]):
                latest_loss = status["var"][phase]["model_metrics"]["agg_loss"][round]
                round_acc = status["var"][phase]["model_metrics"]["agg_loss"].shape[0]
                latest_losses.append(latest_loss)
                rounds.append(round_acc)
                table.add_row(name, str(i + 1), f"{latest_loss:.2f}", f"{round_acc}")
            table.add_row(f"[red]{name}[/]", "all", f"{np.mean(latest_losses):.2f}", f"{np.mean(rounds):.2f}")
            self.output_list.append({"type": "loss", "phase": phase, "name": name, "rep": "all",
                                     "mean": np.mean(latest_losses), "num_rounds": np.mean(rounds)})
        self.console.print(table)

    def print_fairness(self, phase="fit"):
        table = self._init_console_table(column_names=["total", "min sel", "max sel"], title=f"FAIRNESS ({phase})")
        for name, sim in self.sims:
            tot_fairs = []
            min_sels = []
            max_sels = []
            for i, status in enumerate(sim["status"]):
                sel_by_dev = np.sum(status["var"][phase]["devs"]["selected"], axis=0)
                fairness = np.std(sel_by_dev)
                min_sel = np.min(sel_by_dev)
                max_sel = np.max(sel_by_dev)
                tot_fairs.append(fairness)
                min_sels.append(min_sel)
                max_sels.append(max_sel)
                table.add_row(name, str(i + 1), f"{fairness:.2f}", f"{min_sel:d}", f"{max_sel:d}")
            table.add_row(f"[red]{name}[/]", "all", f"{np.mean(tot_fairs):.2f}",
                          f"{np.mean(min_sels):.2f}",  f"{np.mean(max_sels):.2f}")
            self.output_list.append({"type": "fairness", "phase": phase, "name": name, "rep": "all",
                                     "mean": np.mean(tot_fairs), "min": np.mean(min_sels),
                                     "max": np.mean(max_sels)})
        self.console.print(table)

    def close(self):
        self.console.save_html(os.path.join(self.output_dir, 'data.html'), clear=True)
        pd.DataFrame(self.output_list).to_csv(os.path.join(self.output_dir, 'dataset.csv'))
        self.output_report.create_md_file()
