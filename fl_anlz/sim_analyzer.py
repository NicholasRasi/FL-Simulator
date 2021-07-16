import os
import logging
import numpy as np
from json_tricks import load
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter, MaxNLocator
import statistics as stats
from mdutils import MdUtils


class SimAnalyzer:

    def __init__(self,
                 simulations: list,
                 input_dir: str,
                 output_dir: str,
                 logger,
                 extension="pdf",
                 do_report=False,
                 show_plot=False,
                 show_data=False):
        # read file
        self.sims = []
        for sim_name, file in simulations:
            with open(input_dir + "/" + file, 'r') as fp:
                self.sims.append((sim_name, load(fp)))

        self.output_dir = os.path.join(output_dir, '')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if do_report:
            self.ext = ".png"
        else:
            self.ext = "." + extension
        self.show_plot = show_plot
        self.show_data = show_data
        self.logger = logger

        matplotlib_logger = logging.getLogger('matplotlib')
        matplotlib_logger.setLevel(logging.ERROR)

        self.output_report = MdUtils(file_name=output_dir + '/report', title='Simulation Report')


    def _add_img_to_report(self, title, img_filename, level=1):
        self.output_report.new_header(level=level, title=title)
        self.output_report.new_paragraph("![](" + img_filename + ")")

    def _save_show_plot(self, output_filename):
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_filename))
        if self.show_plot:
            plt.show()
        plt.close()

    def _plot_acc_loss(self, phase, color, title, ylabel, legend_loc, key):
        fig, ax = plt.subplots()
        for name, sim in self.sims:
            ys = []
            for status in sim["status"]:
                ys.append(status["var"][phase]["model_metrics"][key])
            y_mean = [np.mean(y) for y in zip(*ys)]
            y_std = [np.std(y) for y in zip(*ys)]
            x = range(1, len(y_mean) + 1)
            ax.errorbar(x, y_mean, fmt='-o', color=color, ecolor=color, yerr=y_std, label=name)

            if self.show_data:
                self.logger.info("{} - agg " + title + ": {}".format(name, ys))

        ax.set(title=title, xlabel="round", ylabel=ylabel)
        ax.legend(loc=legend_loc)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True)

    def plot_accuracy(self, phase="eval", color="r"):
        title = f"Accuracy ({phase})"
        self._plot_acc_loss(phase=phase, color=color, title=title,
                            ylabel="accuracy %", legend_loc=4, key="agg_accuracy")

        output_filename = "agg_accuracy_" + phase + self.ext
        self._save_show_plot(output_filename)
        self._add_img_to_report(title, output_filename)

    def plot_loss(self, phase="eval", color="r"):
        title = f"Loss ({phase})"
        self._plot_acc_loss(phase=phase, color=color, title=title,
                            ylabel="loss", legend_loc=1, key="agg_loss")

        output_filename = "agg_loss_" + phase + self.ext
        self._save_show_plot(output_filename)
        self._add_img_to_report(title, output_filename)

    def _plot_round_times(self, phase, color, title, keys, ylabel="time [s]", legend_loc=1):
        fig, ax = plt.subplots()
        for name, sim in self.sims:
            ys = []
            for status in sim["status"]:
                val = 0
                for key in keys:
                    val += status["var"][phase]["times"][key]
                ys.append(np.amax(val, axis=1))
            y_mean = [np.mean(y) for y in zip(*ys)]
            y_std = [np.std(y) for y in zip(*ys)]
            x = range(1, len(y_mean) + 1)
            ax.errorbar(x, y_mean, fmt='-o', color=color, ecolor=color, yerr=y_std, label=name)
        ax.set(title=title, xlabel='round', ylabel=ylabel)
        ax.legend(loc=legend_loc)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True)

    def plot_computation_time(self, phase="fit", color="k"):
        title = f"Computation Time ({phase})"
        self._plot_round_times(phase=phase, color=color, title=title, keys=["computation"])

        output_filename = "rt_computation_" + phase + self.ext
        self._save_show_plot(output_filename)
        self._add_img_to_report(title, output_filename)

    def plot_communication_time(self, phase="fit", color="k"):
        title = f"Communication Time ({phase})"
        self._plot_round_times(phase=phase, color=color, title=title, keys=["communication"])

        output_filename = "rt_communication_" + phase + self.ext
        self._save_show_plot(output_filename)
        self._add_img_to_report(title, output_filename)

    def plot_total_time(self, phase="fit", color="k"):
        title = f"Total Time ({phase})"
        self._plot_round_times(phase=phase, color=color, title=title, keys=["computation", "communication"])

        output_filename = "rt_total_" + phase + self.ext
        self._save_show_plot(output_filename)
        self._add_img_to_report(title, output_filename)

    def _plot_round_consumptions(self, phase, color, title, key, ylabel, legend_loc=1):
        fig, ax = plt.subplots()
        for name, sim in self.sims:
            ys = []
            for status in sim["status"]:
                val = status["var"][phase]["consumption"][key]
                ys.append(np.sum(val, axis=1))
            y_mean = [np.mean(y) for y in zip(*ys)]
            y_std = [np.std(y) for y in zip(*ys)]
            x = range(1, len(y_mean) + 1)
            ax.errorbar(x, y_mean, fmt='-o', color=color, ecolor=color, yerr=y_std, label=name)

            if self.show_data:
                self.logger.info("{} - " + title + ": \n{}\nagg: {}".format(name, val, y_mean))
        ax.set(title=title, xlabel='round', ylabel=ylabel)
        ax.legend(loc=legend_loc)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True)

    def plot_resources_consumption(self, phase="fit", color="k"):
        title = f"Resources Consumption ({phase})"
        self._plot_round_consumptions(phase=phase, color=color, title=title, key="resources", ylabel="iters")

        output_filename = "consumption_resources_" + phase + self.ext
        self._save_show_plot(output_filename)
        self._add_img_to_report(title, output_filename)

    def plot_network_consumption(self, phase="fit", color="k"):
        title = f"Network Consumption ({phase})"
        self._plot_round_consumptions(phase=phase, color=color, title=title, key="network", ylabel="params")

        output_filename = "consumption_network_" + phase + self.ext
        self._save_show_plot(output_filename)
        self._add_img_to_report(title, output_filename)

    def plot_energy_consumption(self, phase="fit", color="k"):
        title = f"Energy Consumption ({phase})"
        self._plot_round_consumptions(phase=phase, color=color, title=title, key="energy", ylabel="mA")

        output_filename = "consumption_energy_" + phase + self.ext
        self._save_show_plot(output_filename)
        self._add_img_to_report(title, output_filename)

    def plot_available_devices(self, color="k"):
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
            ax.errorbar(x, y_mean, fmt='-o', color=color, yerr=y_std, label=name)
        ax.set(title=title, xlabel='round', ylabel=ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc=legend_loc)
        ax.grid(True)

        output_filename = "devices_available" + self.ext
        self._save_show_plot(output_filename)
        self._add_img_to_report(title, output_filename)

    def plot_selected_devices(self, phase="fit", color="k"):
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
            ax.errorbar(x, y_mean, fmt='-o', color=color, yerr=y_std, label=name)
        ax.set(title=title, xlabel='round', ylabel=ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc=legend_loc)
        ax.grid(True)

        output_filename = "devices_selected_" + phase + self.ext
        self._save_show_plot(output_filename)
        self._add_img_to_report(title, output_filename)

    def plot_available_failed_devices(self, color="k"):
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
            ax.errorbar(x, y_mean, fmt='-o', color=color, yerr=y_std, label=name)
        ax.set(title=title, xlabel='round', ylabel=ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc=legend_loc)
        ax.grid(True)

        output_filename = "devices_available_failed" + self.ext
        self._save_show_plot(output_filename)
        self._add_img_to_report(title, output_filename)

    def plot_selected_successful_devices(self, phase="fit", color="k"):
        title = f"Selected and Successful Devices ({phase})"
        ylabel = "devices"
        legend_loc = 1

        fig, ax = plt.subplots()
        for name, sim in self.sims:
            ys = []
            for status in sim["status"]:
                val = status["var"][phase]["devs"]["selected"] -\
                      (status["var"][phase]["devs"]["selected"] & status["con"]["devs"]["failures"])
                ys.append(np.sum(val, axis=1))
            y_mean = [np.mean(y) for y in zip(*ys)]
            y_std = [np.std(y) for y in zip(*ys)]
            x = range(1, len(y_mean) + 1)
            ax.errorbar(x, y_mean, fmt='-o', color=color, yerr=y_std, label=name)
        ax.set(title=title, xlabel='round', ylabel=ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc=legend_loc)
        ax.grid(True)

        output_filename = "devices_selected_successful_" + phase + self.ext
        self._save_show_plot(output_filename)
        self._add_img_to_report(title, output_filename)

    def _plot_round_configs(self, phase, color, title, key, ylabel, legend_loc=1):
        self.output_report.new_header(level=1, title=title)
        for name, sim in self.sims:
            for i, status in enumerate(sim["status"]):
                fig, ax = plt.subplots()
                val1 = status["var"][phase]["upd_opt_configs"]["global"][key]
                val2 = status["var"][phase]["upd_opt_configs"]["local"][key]
                y1 = np.sum(val1, axis=1)
                y2 = np.sum(val2, axis=1)
                x = range(1, y1.shape[0] + 1)
                ax.plot(x, y1, '--o', color=color, label="$" + name + "_{global}$")
                ax.plot(x, y2, '-o', color=color, label="$" + name + "_{local}$")
                ax.set(title=title, xlabel='round', ylabel=ylabel)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.legend(loc=legend_loc)
                ax.grid()

                output_filename = "config_" + key + "_" + name + "_" + str(i) + "_" + phase + self.ext
                self._save_show_plot(output_filename)
                self._add_img_to_report(str(i), output_filename, level=2)

    def plot_epochs_config(self, phase="fit", color="k"):
        title = f"Epochs ({phase})"
        self._plot_round_configs(phase=phase, color=color, title=title, key="epochs", ylabel="epochs")

    def plot_batch_size_config(self, phase="fit", color="k"):
        title = f"Batch Size ({phase})"
        self._plot_round_configs(phase=phase, color=color, title=title, key="batch_size", ylabel="size")

    def plot_num_examples_config(self, phase="fit", color="k"):
        title = f"Num Examples ({phase})"
        self._plot_round_configs(phase=phase, color=color, title=title, key="num_examples", ylabel="examples")

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

    def plot_devices_bar_availability(self, phase="fit", color="k"):
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
                ax.bar(x, y, color=color, label=name, alpha=1)
                ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.legend(loc=legend_loc)
                ax.grid()

                output_filename = "devs_bar_availability_" + name + "_" + str(i) + "_" + phase + self.ext
                self._save_show_plot(output_filename)
                self._add_img_to_report(str(i), output_filename, level=2)

    def plot_devices_bar_failures(self, phase="fit", color="k"):
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
                ax.bar(x, y, color=color, label=name, alpha=1)
                ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.legend(loc=legend_loc)
                ax.grid()

                output_filename = "devs_bar_failures_" + name + "_" + str(i) + "_" + phase + self.ext
                self._save_show_plot(output_filename)
                self._add_img_to_report(str(i), output_filename, level=2)

    def plot_devices_bar_selected(self, phase="fit", color="k"):
        title = f"Selected ({phase})"
        self.output_report.new_header(level=1, title=title)

        for name, sim in self.sims:
            for i, status in enumerate(sim["status"]):
                fig, ax = plt.subplots()

                # selected
                val = status["var"][phase]["devs"]["selected"]
                y = np.sum(val, axis=0) / val.shape[0]
                x = range(0, y.shape[0])
                ax.bar(x, y, color=color, label=name, alpha=1)
                ax.set(title=title, xlabel="device", ylabel="density")
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.legend(loc=4)
                ax.grid()

                output_filename = "devs_bar_selected_" + name + "_" + str(i) + "_" + phase + self.ext
                self._save_show_plot(output_filename)
                self._add_img_to_report(str(i), output_filename, level=2)

    def plot_devices_data_size(self, color="k"):
        title = "Local Data Size"
        self.output_report.new_header(level=1, title=title)

        for name, sim in self.sims:
            for i, status in enumerate(sim["status"]):
                fig, ax = plt.subplots()

                # local data
                val = status["con"]["devs"]["local_data_sizes"]
                ax.hist(val, color=color, label=name, density=False, alpha=1)
                ax.set(title=title, xlabel='# examples', ylabel='density')
                ax.legend(loc=1)
                ax.grid(True)

                output_filename = "devs_data_size_" + name + "_" + str(i) + self.ext
                self._save_show_plot(output_filename)
                self._add_img_to_report(str(i), output_filename, level=2)

    def plot_devices_ips(self, color="k"):
        title = "IPS"
        self.output_report.new_header(level=1, title=title)

        for name, sim in self.sims:
            for i, status in enumerate(sim["status"]):
                fig, ax = plt.subplots()

                # IPS
                val = status["con"]["devs"]["ips"]
                ax.hist(val, color=color, label=name, density=False, alpha=1)
                ax.set(title=title, xlabel='IPS', ylabel='density')
                ax.legend(loc=1)
                ax.grid(True)

                output_filename = "devs_ips_" + name + "_" + str(i) + self.ext
                self._save_show_plot(output_filename)
                self._add_img_to_report(str(i), output_filename, level=2)

    def plot_devices_available_energy(self, color="k"):
        title = "Energy"
        self.output_report.new_header(level=1, title=title)

        for name, sim in self.sims:
            for i, status in enumerate(sim["status"]):
                fig, ax = plt.subplots()

                # energy
                val = np.mean(status["con"]["devs"]["energy"], axis=1)
                ax.hist(val, color=color,  label=name, density=False, alpha=1)
                ax.set(title=title, xlabel='mAh', ylabel='density')
                ax.legend(loc=1)
                ax.grid(True)

                output_filename = "devs_available_energy_" + name + "_" + str(i) + self.ext
                self._save_show_plot(output_filename)
                self._add_img_to_report(str(i), output_filename, level=2)

    def plot_devices_network_speed(self, color="k"):
        title = "Network Speed"
        self.output_report.new_header(level=1, title=title)

        for name, sim in self.sims:
            for i, status in enumerate(sim["status"]):
                fig, ax = plt.subplots()

                # network
                val = np.mean(status["con"]["devs"]["net_speed"], axis=1)
                ax.hist(val, color=color, label=name, density=False, alpha=1)
                ax.set(title=title, xlabel='params/s', ylabel='density')
                ax.legend(loc=1)
                ax.grid(True)

                output_filename = "devs_network_speed_" + name + "_" + str(i) + self.ext
                self._save_show_plot(output_filename)
                self._add_img_to_report(str(i), output_filename, level=2)

    def plot_devices_data_distribution(self):
        title = "Data Distribution"
        self.output_report.new_header(level=1, title=title)

        for name, sim in self.sims:
            for i, status in enumerate(sim["status"]):
                fig, ax = plt.subplots(1)

                # local data
                local_data_stats = status["con"]["devs"]["local_data_stats"]
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

    def print_data(self):
        np.set_printoptions(precision=2)
        np.set_printoptions(suppress=True)
        for name, sim in self.sims:
            self.logger.info("---- DATA SIMULATION {} ----".format(name))

            for phase in ["fit", "eval"]:
                self.logger.info("--- {} ---".format(phase))
                for key in sim["status"]["var"][phase]:
                    self.logger.info("-- {} --".format(key))
                    for consumption in sim["status"]["var"][phase][key]:
                        self.logger.info("- {} -".format(consumption))
                        val = sim["status"]["var"][phase][key][consumption]

                        self.logger.info("\n{}".format(val))
                        if len(val.shape) > 1:
                            self.logger.info("aggregated: {}".format(np.sum(val, axis=1)))
                        else:
                            self.logger.info("aggregated: {}".format(np.sum(val)))

    def print_metrics(self):
        for name, sim in self.sims:
            tot_tts = []
            mean_rts = []
            tot_rcs = []
            mean_rcs = []
            tot_ecs = []
            mean_ecs = []
            last_accs = []
            last_losses = []

            self.logger.info("\n---- SIMULATION {}".format(name))
            for i, status in enumerate(sim["status"]):
                self.logger.info("--- REP {}".format(i))
                self.logger.info("-- CONSTANT DATA")
                self.logger.info(
                    "\tdev availability: {:.2f} %".format(status["con"]["devs"]["availability"].mean() * 100))
                self.logger.info("\tdev failures: {:.2f} %".format(status["con"]["devs"]["failures"].mean() * 100))
                self.logger.info("\tips [IPS], mean: {:.2f} | std: {:.2f} ".format(status["con"]["devs"]["ips"].mean(),
                                                                                   status["con"]["devs"]["ips"].std()))
                self.logger.info(
                    "\tenergy [mAh], mean: {:.2f} | std: {:.2f}".format(status["con"]["devs"]["energy"].mean(),
                                                                        status["con"]["devs"]["energy"].std()))
                self.logger.info("\tnet speed [params/s], mean: {:.2f} | std: {:.2f}".format(
                    status["con"]["devs"]["net_speed"].mean(), status["con"]["devs"]["net_speed"].std()))
                self.logger.info("\tlocal data size [examples], mean {:.2f} | std: {:.2f}".format(
                    status["con"]["devs"]["local_data_sizes"].mean(), status["con"]["devs"]["local_data_sizes"].std()))
                self.logger.info('\tmodel params: {}'.format(status["con"]["model"]["tot_weights"]))
                self.logger.info("-- VARIABLE DATA")
                self.logger.info("- SELECTION")

                rt = status["var"]["fit"]["times"]["computation"] + status["var"]["fit"]["times"]["communication"]
                tot_tt = np.sum(np.amax(rt, axis=1))
                mean_rt = np.mean(np.amax(rt, axis=1))
                rc = status["var"]["fit"]["consumption"]["resources"]
                tot_rc = np.sum(rc)
                mean_rc = np.mean(np.sum(rc, axis=1))
                ec = status["var"]["fit"]["consumption"]["energy"]
                tot_ec = np.sum(ec)
                mean_ec = np.mean(np.sum(ec, axis=1))
                last_acc = status["var"]["eval"]["model_metrics"]["agg_accuracy"][-1]
                last_loss = status["var"]["eval"]["model_metrics"]["agg_loss"][-1]

                tot_tts.append(tot_tt)
                mean_rts.append(mean_rt)
                tot_rcs.append(tot_rc)
                mean_rcs.append(mean_rc)
                tot_ecs.append(tot_ec)
                mean_ecs.append(mean_ec)
                last_accs.append(last_acc)
                last_losses.append(last_loss)

                self.logger.info("\tmean failed devices (among all devices): {:.2f} %".format(
                    status["con"]["devs"]["failures"].mean() * 100))
                self.logger.info("\tmean available devices (among all devices): {:.2f} %".format(
                    status["con"]["devs"]["availability"].mean() * 100))
                self.logger.info(
                    "\tmean selected devices: {:.2f} %".format(status["var"]["fit"]["devs"]["selected"].mean() * 100))
                successful_devs = status["var"]["fit"]["devs"]["selected"] - (
                        status["var"]["fit"]["devs"]["selected"] & status["con"]["devs"]["failures"])
                self.logger.info("\tmean successful devices (among selected ones): {:.2f} %".format(
                    successful_devs.sum() / status["var"]["fit"]["devs"]["selected"].sum() * 100))
                self.logger.info("\tmean successful devices (among selected ones): {:.2f} %".format(
                    successful_devs.sum() / status["var"]["fit"]["devs"]["selected"].sum() * 100))
                self.logger.info("- TIMES")
                self.logger.info("\ttraining time [s], tot: {:.2f} | avg round: {:.2f}".format(tot_tt, mean_rt))
                self.logger.info("- CONSUMPTIONS")
                self.logger.info("\tresources [iters], tot: {:.2f} | avg round: {:.2f}".format(tot_rc, mean_rc))
                self.logger.info("\tenergy [mAh], tot: {:.2f} | avg round: {:.2f}".format(tot_ec, mean_ec))
                self.logger.info("- ACCURACY")
                self.logger.info("\taccuracy [%], last: {:.2f}".format(last_acc))
                self.logger.info("\tloss [%], last: {:.2f}".format(last_loss))

            self.logger.info("- AGGREGATED")
            self.logger.info("\taccuracy [%], last mean: {:.2f} | std: {:.2f}".format(stats.mean(last_accs),
                                                                                      stats.stdev(last_accs) if len(
                                                                                          last_accs) > 1 else 0))
            self.logger.info("\tloss, last mean: {:.2f} | std: {:.2f}".format(stats.mean(last_losses),
                                                                              stats.stdev(last_losses) if len(
                                                                                  last_losses) > 1 else 0))
            self.logger.info("\ttraining time [s], tot: {:.2f} | avg round: {:.2f}".format(stats.mean(tot_tts),
                                                                                           stats.mean(mean_rts)))
            self.logger.info("\tresources [iters], tot: {:.2f} | avg round: {:.2f}".format(stats.mean(tot_rcs),
                                                                                           stats.mean(mean_rcs)))
            self.logger.info(
                "\tenergy [mAh], tot: {:.2f} | avg round: {:.2f}".format(stats.mean(tot_ecs), stats.mean(mean_ecs)))

    def close(self):
        self.output_report.create_md_file()
