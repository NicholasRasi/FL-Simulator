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

        self.ext = "." + extension
        self.show_plot = show_plot
        self.show_data = show_data
        self.logger = logger

        matplotlib_logger = logging.getLogger('matplotlib')
        matplotlib_logger.setLevel(logging.ERROR)

        self.output_report = MdUtils(file_name=output_dir + '/report', title='Simulation Report')

    def plot_agg_accuracy_loss(self, phase="eval"):
        """
        Plot the aggregated accuracy and loss.
        If there are multiple repetitions (simulations) the values are averaged
        and the standard deviation is reported in the graph.
        :param phase:
        :return:
        """
        fig, ax = plt.subplots(2)

        def plot(subplot, title, color, ylabel, legend_loc, metrics):
            for name, sim in self.sims:
                ys = []
                for status in sim["status"]:
                    ys.append(status["var"][phase]["model_metrics"][metrics])
                y_mean = [np.mean(y) for y in zip(*ys)]
                y_std = [np.std(y) for y in zip(*ys)]
                x = range(1, len(y_mean)+1)
                ax[subplot].errorbar(x, y_mean, fmt='-o', color=color, ecolor=color, yerr=y_std, label=name)

                if self.show_data:
                    self.logger.info("{} - agg " + title + ": {}".format(name, ys))
            ax[subplot].set(title=title + ' ('+phase+')', xlabel='round', ylabel=ylabel)
            ax[subplot].legend(loc=legend_loc)
            ax[subplot].xaxis.set_major_locator(MaxNLocator(integer=True))
            # ax[0].xaxis.set_ticks(x[0::1])
            ax[subplot].grid()

        plot(0, "Accuracy", "r", "accuracy %", 4, "agg_accuracy")
        ax[0].yaxis.set_major_formatter(PercentFormatter(xmax=1))
        plot(1, "Loss", "b", "loss", 1, "agg_loss")

        output_filename = "agg_accuracy_loss_" + phase + self.ext
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_filename))
        if self.show_plot:
            plt.show()
        plt.close()

        self.output_report.new_header(level=1, title='Accuracy and Loss')
        self.output_report.new_paragraph("![](" + output_filename + ")")

    def plot_round_times(self, phase="fit"):
        fig, ax = plt.subplots(3)

        def plot(subplot, title, color, times):
            for name, sim in self.sims:
                ys = []
                for status in sim["status"]:
                    val = 0
                    for time in times:
                        val += status["var"][phase]["times"][time]
                    ys.append(np.amax(val, axis=1))
                y_mean = [np.mean(y) for y in zip(*ys)]
                y_std = [np.std(y) for y in zip(*ys)]
                x = range(1, len(y_mean) + 1)
                ax[subplot].errorbar(x, y_mean, fmt='-o', color=color, ecolor=color, yerr=y_std, label=name)
            ax[subplot].set(title=title + ' (' + phase + ')', xlabel='round', ylabel='time [s]')
            ax[subplot].legend(loc=1)
            ax[subplot].xaxis.set_major_locator(MaxNLocator(integer=True))
            #ax[subplot].xaxis.set_ticks(x[0::1])
            ax[subplot].grid()

        plot(0, "Computation Time", "b", ["computation"])
        plot(1, "Communication Time", "g", ["communication"])
        plot(2, "Total Time", "r", ["computation", "communication"])

        output_filename = "round_times_" + phase + self.ext
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_filename))
        if self.show_plot:
            plt.show()
        plt.close()

        self.output_report.new_header(level=1, title='Round times')
        self.output_report.new_paragraph("![](" + output_filename + ")")


    def plot_round_consumptions(self, phase="fit"):
        fig, ax = plt.subplots(3)

        def plot(subplot, title, color, ylabel, consumption):
            for name, sim in self.sims:
                ys = []
                for status in sim["status"]:
                    val = status["var"][phase]["consumption"][consumption]
                    ys.append(np.sum(val, axis=1))
                y_mean = [np.mean(y) for y in zip(*ys)]
                y_std = [np.std(y) for y in zip(*ys)]
                x = range(1, len(y_mean) + 1)
                ax[subplot].errorbar(x, y_mean, fmt='-o', color=color, ecolor=color, yerr=y_std, label=name)

                if self.show_data:
                    self.logger.info("{} - " + title + ": \n{}\nagg: {}".format(name, val, y_mean))
            ax[subplot].set(title=title + ' (' + phase + ')', xlabel='round', ylabel=ylabel)
            ax[subplot].legend(loc=1)
            ax[subplot].xaxis.set_major_locator(MaxNLocator(integer=True))
            #ax[subplot].xaxis.set_ticks(x[0::1])
            ax[subplot].grid()

        plot(0, "Resource Consumption", "b", "iterations", "resources")
        plot(1, "Network Consumption", "g", "params", "network")
        plot(2, "Energy Consumption", "c", "mA", "energy")

        plt.tight_layout()
        output_filename = "round_consumptions_" + phase + self.ext
        plt.savefig(os.path.join(self.output_dir, output_filename))
        if self.show_plot:
            plt.show()
        plt.close()

        self.output_report.new_header(level=1, title='Consumptions')
        self.output_report.new_paragraph("![](" + output_filename + ")")

    def plot_round_devices(self, phase="fit"):
        fig, ax = plt.subplots(4)

        # available devices
        for name, sim in self.sims:
            ys = []
            for status in sim["status"]:
                val = status["con"]["devs"]["availability"]
                ys.append(np.sum(val, axis=1))
            y_mean = [np.mean(y) for y in zip(*ys)]
            y_std = [np.std(y) for y in zip(*ys)]
            x = range(1, len(y_mean) + 1)
            ax[0].errorbar(x, y_mean, fmt='-o', color="b", yerr=y_std, label=name)
        ax[0].set(title='Available Devices', xlabel='round', ylabel='devices')
        ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        #ax[0].xaxis.set_ticks(x[0::1])
        ax[0].legend(loc=1)
        ax[0].grid()

        # selected devices
        for name, sim in self.sims:
            ys = []
            for status in sim["status"]:
                val = status["var"][phase]["devs"]["selected"]
                ys.append(np.sum(val, axis=1))
            y_mean = [np.mean(y) for y in zip(*ys)]
            y_std = [np.std(y) for y in zip(*ys)]
            x = range(1, len(y_mean) + 1)
            ax[1].errorbar(x, y_mean, fmt='-o', color="g", yerr=y_std, label=name)
        ax[1].set(title='Selected Devices (' + phase + ')', xlabel='round', ylabel='devices')
        ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        #ax[1].xaxis.set_ticks(x[0::1])
        ax[1].legend(loc=1)
        ax[1].grid()

        # available and failed devices
        for name, sim in self.sims:
            ys = []
            for status in sim["status"]:
                val = status["con"]["devs"]["availability"] & status["con"]["devs"]["failures"]
                ys.append(np.sum(val, axis=1))
            y_mean = [np.mean(y) for y in zip(*ys)]
            y_std = [np.std(y) for y in zip(*ys)]
            x = range(1, len(y_mean) + 1)
            ax[2].errorbar(x, y_mean, fmt='-o', color="c", yerr=y_std, label=name)
        ax[2].set(title='Available & Failed Devices', xlabel='round', ylabel='devices')
        ax[2].xaxis.set_major_locator(MaxNLocator(integer=True))
        #ax[2].xaxis.set_ticks(x[0::1])
        ax[2].legend(loc=1)
        ax[2].grid()

        # selected and successful
        for name, sim in self.sims:
            ys = []
            for status in sim["status"]:
                val = status["var"][phase]["devs"]["selected"] - (status["var"][phase]["devs"]["selected"] & status["con"]["devs"]["failures"])
                ys.append(np.sum(val, axis=1))
            y_mean = [np.mean(y) for y in zip(*ys)]
            y_std = [np.std(y) for y in zip(*ys)]
            x = range(1, len(y_mean) + 1)
            ax[3].errorbar(x, y_mean, fmt='-o', color="m", yerr=y_std, label=name)
        ax[3].set(title='Selected & Successful Devices (' + phase + ')', xlabel='round', ylabel='devices')
        ax[3].xaxis.set_major_locator(MaxNLocator(integer=True))
        # ax[3].xaxis.set_ticks(x[0::1])
        ax[3].legend(loc=1)
        ax[3].grid()

        plt.tight_layout()
        output_filename = "round_devices_" + phase + self.ext
        plt.savefig(os.path.join(self.output_dir, output_filename))
        if self.show_plot:
            plt.show()
        plt.close()

        self.output_report.new_header(level=1, title='Devices')
        self.output_report.new_paragraph("![](" + output_filename + ")")

    def plot_round_configs(self, phase="fit"):
        self.output_report.new_header(level=1, title='Devices')

        for name, sim in self.sims:
            for i, status in enumerate(sim["status"]):
                fig, ax = plt.subplots(3)

                def plot(subplot, title, color, ylabel, config):
                    val1 = status["var"][phase]["upd_opt_configs"]["global"][config]
                    val2 = status["var"][phase]["upd_opt_configs"]["local"][config]
                    y1 = np.sum(val1, axis=1)
                    y2 = np.sum(val2, axis=1)
                    x = range(1, y1.shape[0] + 1)
                    ax[subplot].plot(x, y1, '--o', color=color, label="$" + name + "_{global}$")
                    ax[subplot].plot(x, y2, '-o', color=color, label="$" + name + "_{local}$")
                    ax[subplot].set(title=title + ' (' + phase + ')', xlabel='round', ylabel=ylabel)
                    ax[subplot].xaxis.set_major_locator(MaxNLocator(integer=True))
                    # ax[subplot].xaxis.set_ticks(x[0::1])
                    ax[subplot].legend(loc=4)
                    ax[subplot].grid()

                plot(0, "Epochs", "b", "epochs", "epochs")
                plot(1, "Batch size", "g", "size", "batch_size")
                plot(2, "Num Examples", "c", "examples", "num_examples")

                plt.tight_layout()
                output_filename = "round_configs_" + name + "_" + str(i) + "_" + phase + self.ext
                plt.savefig(os.path.join(self.output_dir, output_filename))
                if self.show_plot:
                    plt.show()
                plt.close()

                self.output_report.new_header(level=2, title=name + " " + str(i) + " " + phase)
                self.output_report.new_paragraph("![](" + output_filename + ")")

    def plot_matrix_devices(self, phase="fit"):
        self.output_report.new_header(level=1, title='Devices Matrix')

        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        cm = LinearSegmentedColormap.from_list("dev_cmap", colors)
        values = [0, 1, 2]
        labels = ["Not available", "Available", "Selected"]

        for name, sim in self.sims:
            for j, status in enumerate(sim["status"]):
                fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False)
                i = 0
                ax[i][0].set(title='Selected Devices ' + name + ' (' + phase + ')', xlabel='devices', ylabel='round')
                im = ax[i][0].imshow(status["con"]["devs"]["availability"].T + status["var"][phase]["devs"]["selected"].T, interpolation=None, cmap=cm)
                i += 1
                colors = [im.cmap(im.norm(value)) for value in values]
                patches = [mpatches.Patch(color=colors[i], label=labels[i].format(l=values[i])) for i in range(len(values))]
                plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

                plt.tight_layout()
                output_filename = "matrix_devices_" + name + "_" + str(j) + "_" + phase + self.ext
                plt.savefig(os.path.join(self.output_dir, output_filename))
                if self.show_plot:
                    plt.show()
                plt.close()

                self.output_report.new_header(level=2, title=name + " " + str(i) + " " + phase)
                self.output_report.new_paragraph("![](" + output_filename + ")")

    def plot_devices_bar(self, phase="fit"):
        self.output_report.new_header(level=1, title='Devices Bar')

        for name, sim in self.sims:
            for i, status in enumerate(sim["status"]):
                fig, ax = plt.subplots(3)

                # availability
                r = 0
                val = status["con"]["devs"]["availability"]
                y = np.sum(val, axis=0)/val.shape[0]
                x = range(0, y.shape[0])
                ax[r].bar(x, y, label=name, alpha=1)
                ax[r].set(title='Availability', xlabel='device', ylabel='density')
                ax[r].yaxis.set_major_formatter(PercentFormatter(xmax=1))
                ax[r].xaxis.set_major_locator(MaxNLocator(integer=True))
                ax[r].legend(loc=1)
                ax[r].grid()

                # failures
                r += 1
                val = status["con"]["devs"]["failures"]
                y = np.sum(val, axis=0) / val.shape[0]
                x = range(0, y.shape[0])
                ax[r].bar(x, y, label=name, alpha=1)
                ax[r].set(title='Failures', xlabel='device', ylabel='density')
                ax[r].yaxis.set_major_formatter(PercentFormatter(xmax=1))
                ax[r].xaxis.set_major_locator(MaxNLocator(integer=True))
                ax[r].legend(loc=1)
                ax[r].grid()

                # selected
                r += 1
                val = status["var"][phase]["devs"]["selected"]
                y = np.sum(val, axis=0) / val.shape[0]
                x = range(0, y.shape[0])
                ax[r].bar(x, y, label=name, alpha=1)
                ax[r].set(title='Selected', xlabel='device', ylabel='density')
                ax[r].yaxis.set_major_formatter(PercentFormatter(xmax=1))
                ax[r].xaxis.set_major_locator(MaxNLocator(integer=True))
                ax[r].legend(loc=1)
                ax[r].grid()

                plt.tight_layout()
                output_filename = "devs_bar_" + name + "_" + str(i) + "_" + phase + self.ext
                plt.savefig(os.path.join(self.output_dir, output_filename))
                if self.show_plot:
                    plt.show()
                plt.close()

                self.output_report.new_header(level=2, title=name + " " + str(i) + " " + phase)
                self.output_report.new_paragraph("![](" + output_filename + ")")

    def plot_devices_capabilities_bar(self):
        self.output_report.new_header(level=1, title='Capabilities Bar')

        for name, sim in self.sims:
            for i, status in enumerate(sim["status"]):
                fig, ax = plt.subplots(3)

                # availability
                r = 0
                y = status["con"]["devs"]["ips"]
                x = range(0, y.shape[0])
                ax[r].bar(x, y, label=name, alpha=0.5)
                ax[r].set(title='Computation Speed', xlabel='device', ylabel='IPS')
                ax[r].xaxis.set_major_locator(MaxNLocator(integer=True))
                ax[r].legend(loc=1)
                ax[r].grid()

                # energy
                r += 1
                val = status["con"]["devs"]["energy"]
                y = np.mean(val, axis=0)
                x = range(0, y.shape[0])
                ax[r].bar(x, y, label=name, alpha=0.5)
                ax[r].set(title='Available Energy', xlabel='device', ylabel='mAh')
                ax[r].xaxis.set_major_locator(MaxNLocator(integer=True))
                ax[r].legend(loc=1)
                ax[r].grid()

                # network
                r += 1
                val = status["con"]["devs"]["net_speed"]
                y = np.mean(val, axis=0)
                x = range(0, y.shape[0])
                ax[r].bar(x, y, label=name, alpha=0.5)
                ax[r].set(title='Network Speed', xlabel='device', ylabel='params/s')
                ax[r].xaxis.set_major_locator(MaxNLocator(integer=True))
                ax[r].legend(loc=1)
                ax[r].grid()

                plt.tight_layout()
                output_filename = "devs_capabilities_bar_" + name + "_" + str(i) + self.ext
                plt.savefig(os.path.join(self.output_dir, output_filename))
                if self.show_plot:
                    plt.show()
                plt.close()

                self.output_report.new_header(level=2, title=name + " " + str(i))
                self.output_report.new_paragraph("![](" + output_filename + ")")

    def plot_devs_data(self):
        self.output_report.new_header(level=1, title='Devs Data')

        for name, sim in self.sims:
            for i, status in enumerate(sim["status"]):
                fig, ax = plt.subplots(4)

                # local data
                val = status["con"]["devs"]["local_data_sizes"]
                ax[0].hist(val, label=name, density=False, alpha=1)
                ax[0].set(title='Local Data Size', xlabel='# examples', ylabel='density')
                #ax[0].yaxis.set_major_formatter(PercentFormatter(xmax=1))
                ax[0].legend(loc=1)
                ax[0].grid()

                # IPS
                val = status["con"]["devs"]["ips"]
                ax[1].hist(val, label=name, density=False, alpha=1)
                ax[1].set(title='Computation Speed', xlabel='IPS', ylabel='density')
                # ax[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
                ax[1].legend(loc=1)
                ax[1].grid()

                # energy
                val = np.mean(status["con"]["devs"]["energy"], axis=1)
                ax[2].hist(val, label=name, density=False, alpha=1)
                ax[2].set(title='Available Energy', xlabel='mAh', ylabel='density')
                # ax[2].yaxis.set_major_formatter(PercentFormatter(xmax=1))
                ax[2].legend(loc=1)
                ax[2].grid()

                # network
                val = np.mean(status["con"]["devs"]["net_speed"], axis=1)
                ax[3].hist(val, label=name, density=False, alpha=1)
                ax[3].set(title='Network Speed', xlabel='params/s', ylabel='density')
                # ax[3].yaxis.set_major_formatter(PercentFormatter(xmax=1))
                ax[3].legend(loc=1)
                ax[3].grid()

                plt.tight_layout()
                output_filename = "devs_data_" + name + "_" + str(i) + self.ext
                plt.savefig(os.path.join(self.output_dir, output_filename))
                if self.show_plot:
                    plt.show()
                plt.close()

                self.output_report.new_header(level=2, title=name + " " + str(i))
                self.output_report.new_paragraph("![](" + output_filename + ")")

    def plot_devs_local_data(self):
        self.output_report.new_header(level=1, title='Devs Local Data')

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
                ax.set(title='Data Distribution', xlabel='#', ylabel='device')

                plt.tight_layout()
                output_filename = "devs_local_data_" + name + "_" + str(i) + self.ext
                plt.savefig(os.path.join(self.output_dir, output_filename))
                if self.show_plot:
                    plt.show()
                plt.close()

                self.output_report.new_header(level=2, title=name + " " + str(i))
                self.output_report.new_paragraph("![](" + output_filename + ")")

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
                self.logger.info("\tdev availability: {:.2f} %".format(status["con"]["devs"]["availability"].mean() * 100))
                self.logger.info("\tdev failures: {:.2f} %".format(status["con"]["devs"]["failures"].mean() * 100))
                self.logger.info("\tips [IPS], mean: {:.2f} | std: {:.2f} ".format(status["con"]["devs"]["ips"].mean(), status["con"]["devs"]["ips"].std()))
                self.logger.info("\tenergy [mAh], mean: {:.2f} | std: {:.2f}".format(status["con"]["devs"]["energy"].mean(), status["con"]["devs"]["energy"].std()))
                self.logger.info("\tnet speed [params/s], mean: {:.2f} | std: {:.2f}".format(status["con"]["devs"]["net_speed"].mean(), status["con"]["devs"]["net_speed"].std()))
                self.logger.info("\tlocal data size [examples], mean {:.2f} | std: {:.2f}".format(status["con"]["devs"]["local_data_sizes"].mean(), status["con"]["devs"]["local_data_sizes"].std()))
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

                self.logger.info("\tmean failed devices (among all devices): {:.2f} %".format(status["con"]["devs"]["failures"].mean() * 100))
                self.logger.info("\tmean available devices (among all devices): {:.2f} %".format(status["con"]["devs"]["availability"].mean() * 100))
                self.logger.info("\tmean selected devices: {:.2f} %".format(status["var"]["fit"]["devs"]["selected"].mean() * 100))
                successful_devs = status["var"]["fit"]["devs"]["selected"] - (status["var"]["fit"]["devs"]["selected"] & status["con"]["devs"]["failures"])
                self.logger.info("\tmean successful devices (among selected ones): {:.2f} %".format(successful_devs.sum() / status["var"]["fit"]["devs"]["selected"].sum() * 100))
                self.logger.info("\tmean successful devices (among selected ones): {:.2f} %".format(successful_devs.sum() / status["var"]["fit"]["devs"]["selected"].sum() * 100))
                self.logger.info("- TIMES")
                self.logger.info("\ttraining time [s], tot: {:.2f} | avg round: {:.2f}".format(tot_tt, mean_rt))
                self.logger.info("- CONSUMPTIONS")
                self.logger.info("\tresources [iters], tot: {:.2f} | avg round: {:.2f}".format(tot_rc, mean_rc))
                self.logger.info("\tenergy [mAh], tot: {:.2f} | avg round: {:.2f}".format(tot_ec, mean_ec))
                self.logger.info("- ACCURACY")
                self.logger.info("\taccuracy [%], last: {:.2f}".format(last_acc))
                self.logger.info("\tloss [%], last: {:.2f}".format(last_loss))

            self.logger.info("- AGGREGATED")
            self.logger.info("\taccuracy [%], last mean: {:.2f} | std: {:.2f}".format(stats.mean(last_accs), stats.stdev(last_accs) if len(last_accs) > 1 else 0))
            self.logger.info("\tloss, last mean: {:.2f} | std: {:.2f}".format(stats.mean(last_losses), stats.stdev(last_losses) if len(last_losses) > 1 else 0))
            self.logger.info("\ttraining time [s], tot: {:.2f} | avg round: {:.2f}".format(stats.mean(tot_tts), stats.mean(mean_rts)))
            self.logger.info("\tresources [iters], tot: {:.2f} | avg round: {:.2f}".format(stats.mean(tot_rcs), stats.mean(mean_rcs)))
            self.logger.info("\tenergy [mAh], tot: {:.2f} | avg round: {:.2f}".format(stats.mean(tot_ecs), stats.mean(mean_ecs)))


    def close(self):
        self.output_report.create_md_file()
