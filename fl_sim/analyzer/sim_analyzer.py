import os
import logging
import numpy as np
from json_tricks import load
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
from pylab import rcParams
import statistics as stats


class SimAnalyzer:

    def __init__(self, simulations: list, input_dir: str, output_dir: str, logger, extension="pdf", show_plot=False, show_data=False):
        # read file
        self.sims = []
        for sim_name, file in simulations:
            with open(input_dir + "/" + file, 'r') as fp:
                self.sims.append((sim_name, load(fp)))

        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.ext = "." + extension
        self.show_plot = show_plot
        self.show_data = show_data
        self.logger = logger

        matplotlib_logger = logging.getLogger('matplotlib')
        matplotlib_logger.setLevel(logging.ERROR)

    def plot_agg_accuracy_loss(self, phase="eval"):
        fig, ax = plt.subplots(2)

        # plot accuracy
        for name, sim in self.sims:
            ys = []
            for status in sim["status"]:
                ys.append(status["var"][phase]["model_metrics"]["agg_accuracy"])
            y_mean = [np.mean(y) for y in zip(*ys)]
            y_std = [np.std(y) for y in zip(*ys)]
            x = range(1, len(y_mean)+1)
            ax[0].errorbar(x, y_mean, fmt='-o', yerr=y_std, label=name)

            if self.show_data:
                self.logger.info("{} - agg accuracy: {}".format(name, y))
        ax[0].set(title='Accuracy ('+phase+')', xlabel='round', ylabel='accuracy %')
        ax[0].legend(loc=4)
        # ax[0].xaxis.set_ticks(x[0::1])
        ax[0].yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax[0].grid()

        # plot loss
        for name, sim in self.sims:
            ys = []
            for status in sim["status"]:
                ys.append(status["var"][phase]["model_metrics"]["agg_loss"])
            y_mean = [np.mean(y) for y in zip(*ys)]
            y_std = [np.std(y) for y in zip(*ys)]
            x = range(1, len(y_mean) + 1)
            ax[1].errorbar(x, y_mean, fmt='-o', yerr=y_std, label=name)

            if self.show_data:
                self.logger.info("{} - agg loss: {}".format(name, y))
        ax[1].set(title='Loss ('+phase+')', xlabel='round', ylabel='loss')
        ax[1].legend(loc=1)
        # ax[1].xaxis.set_ticks(x[0::1])
        ax[1].grid()

        plt.tight_layout()
        plt.savefig(self.output_dir + "/agg_accuracy_loss_" + phase + self.ext)
        if self.show_plot:
            plt.show()
        plt.close()

    def plot_round_times(self, phase="fit"):
        fig, ax = plt.subplots(3)

        # computation
        for name, sim in self.sims:
            ys = []
            for status in sim["status"]:
                val = status["var"][phase]["times"]["computation"]
                ys.append(np.amax(val, axis=1))
            y_mean = [np.mean(y) for y in zip(*ys)]
            y_std = [np.std(y) for y in zip(*ys)]
            x = range(1, len(y_mean) + 1)
            ax[0].errorbar(x, y_mean, fmt='-o', yerr=y_std, label=name)
        ax[0].set(title='Computation Time (' + phase + ')', xlabel='round', ylabel='time [s]')
        ax[0].legend(loc=1)
        ax[0].xaxis.set_ticks(x[0::1])
        ax[0].grid()

        # communication
        for name, sim in self.sims:
            ys = []
            for status in sim["status"]:
                val = status["var"][phase]["times"]["communication"]
                ys.append(np.amax(val, axis=1))
            y_mean = [np.mean(y) for y in zip(*ys)]
            y_std = [np.std(y) for y in zip(*ys)]
            x = range(1, len(y_mean) + 1)
            ax[1].errorbar(x, y_mean, fmt='-o', yerr=y_std, label=name)
        ax[1].set(title='Communication Time (' + phase + ')', xlabel='round', ylabel='time [s]')
        ax[1].legend(loc=1)
        ax[1].xaxis.set_ticks(x[0::1])
        ax[1].grid()

        # round time = computation + communication
        for name, sim in self.sims:
            ys = []
            for status in sim["status"]:
                val = status["var"]["fit"]["times"]["computation"] + status["var"]["fit"]["times"]["communication"]
                ys.append(np.amax(val, axis=1))
            y_mean = [np.mean(y) for y in zip(*ys)]
            y_std = [np.std(y) for y in zip(*ys)]
            x = range(1, len(y_mean) + 1)
            ax[2].errorbar(x, y_mean, fmt='-o', yerr=y_std, label=name)
        ax[2].set(title='Total Time (' + phase + ')', xlabel='round', ylabel='time [s]')
        ax[2].legend(loc=1)
        ax[2].xaxis.set_ticks(x[0::1])
        ax[2].grid()

        plt.tight_layout()
        plt.savefig(self.output_dir + "/round_times_" + phase + self.ext)
        if self.show_plot:
            plt.show()
        plt.close()


    def plot_round_consumptions(self, phase="fit"):
        fig, ax = plt.subplots(3)

        # resources consumption
        for name, sim in self.sims:
            ys = []
            for status in sim["status"]:
                val = status["var"][phase]["consumption"]["resources"]
                ys.append(np.sum(val, axis=1))
            y_mean = [np.mean(y) for y in zip(*ys)]
            y_std = [np.std(y) for y in zip(*ys)]
            x = range(1, len(y_mean) + 1)
            ax[0].errorbar(x, y_mean, fmt='-o', yerr=y_std, label=name)

            if self.show_data:
                self.logger.info("{} - resources consumption: \n{}\nagg: {}".format(name, val, y))
        ax[0].set(title='Resource Consumption ('+phase+')', xlabel='round', ylabel='iterations')
        ax[0].legend(loc=1)
        ax[0].xaxis.set_ticks(x[0::1])
        ax[0].grid()

        # network consumption
        for name, sim in self.sims:
            ys = []
            for status in sim["status"]:
                val = status["var"][phase]["consumption"]["network"]
                ys.append(np.sum(val, axis=1))
            y_mean = [np.mean(y) for y in zip(*ys)]
            y_std = [np.std(y) for y in zip(*ys)]
            x = range(1, len(y_mean) + 1)
            ax[1].errorbar(x, y_mean, fmt='-o', yerr=y_std, label=name)
        ax[1].set(title='Network Consumption ('+phase+')', xlabel='round', ylabel='params')
        ax[1].legend(loc=1)
        ax[1].xaxis.set_ticks(x[0::1])
        ax[1].grid()

        # energy consumption
        for name, sim in self.sims:
            ys = []
            for status in sim["status"]:
                val = status["var"][phase]["consumption"]["energy"]
                ys.append(np.sum(val, axis=1))
            y_mean = [np.mean(y) for y in zip(*ys)]
            y_std = [np.std(y) for y in zip(*ys)]
            x = range(1, len(y_mean) + 1)
            ax[2].errorbar(x, y_mean, fmt='-o', yerr=y_std, label=name)
        ax[2].set(title='Energy Consumption ('+phase+')', xlabel='round', ylabel='mA')
        ax[2].legend(loc=1)
        ax[2].xaxis.set_ticks(x[0::1])
        ax[2].grid()

        plt.tight_layout()
        plt.savefig(self.output_dir + "/round_consumptions_" + phase + self.ext)
        if self.show_plot:
            plt.show()
        plt.close()

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
            ax[0].errorbar(x, y_mean, fmt='-o', yerr=y_std, label=name)
        ax[0].set(title='Available Devices', xlabel='round', ylabel='devices')
        ax[0].xaxis.set_ticks(x[0::1])
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
            ax[1].errorbar(x, y_mean, fmt='-o', yerr=y_std, label=name)
        ax[1].set(title='Selected Devices (' + phase + ')', xlabel='round', ylabel='devices')
        ax[1].xaxis.set_ticks(x[0::1])
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
            ax[2].errorbar(x, y_mean, fmt='-o', yerr=y_std, label=name)
        ax[2].set(title='Available & Failed Devices', xlabel='round', ylabel='devices')
        ax[2].xaxis.set_ticks(x[0::1])
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
            ax[3].errorbar(x, y_mean, fmt='-o', yerr=y_std, label=name)
        ax[3].set(title='Selected & Successful Devices (' + phase + ')', xlabel='round', ylabel='devices')
        ax[3].xaxis.set_ticks(x[0::1])
        ax[3].legend(loc=1)
        ax[3].grid()

        plt.tight_layout()
        plt.savefig(self.output_dir + "/round_devices_" + phase + self.ext)
        if self.show_plot:
            plt.show()
        plt.close()

    def plot_round_configs(self, phase="fit"):
        for name, sim in self.sims:
            for i, status in enumerate(sim["status"]):
                fig, ax = plt.subplots(3)
                # epochs
                val1 = status["var"][phase]["upd_opt_configs"]["global"]["epochs"]
                val2 = status["var"][phase]["upd_opt_configs"]["local"]["epochs"]
                y1 = np.sum(val1, axis=1)
                y2 = np.sum(val2, axis=1)
                x = range(1, y1.shape[0] + 1)
                ax[0].plot(x, y1, '--o', label="$"+name+"_{global}$")
                ax[0].plot(x, y2, '-o', label="$"+name+"_{local}$")
                ax[0].set(title='Epochs (' + phase + ')', xlabel='round', ylabel='epochs')
                ax[0].xaxis.set_ticks(x[0::1])
                ax[0].legend(loc=4)
                ax[0].grid()

                # batch size
                val1 = status["var"][phase]["upd_opt_configs"]["global"]["batch_size"]
                val2 = status["var"][phase]["upd_opt_configs"]["local"]["batch_size"]
                y1 = np.sum(val1, axis=1)
                y2 = np.sum(val2, axis=1)
                x = range(1, y1.shape[0] + 1)
                ax[1].plot(x, y1, '--o', label="$" + name + "_{global}$")
                ax[1].plot(x, y2, '-o', label="$" + name + "_{local}$")
                ax[1].set(title='Batch size (' + phase + ')', xlabel='round', ylabel='size')
                ax[1].xaxis.set_ticks(x[0::1])
                ax[1].legend(loc=4)
                ax[1].grid()

                # num examples
                val1 = status["var"][phase]["upd_opt_configs"]["global"]["num_examples"]
                val2 = status["var"][phase]["upd_opt_configs"]["local"]["num_examples"]
                y1 = np.sum(val1, axis=1)
                y2 = np.sum(val2, axis=1)
                x = range(1, y1.shape[0] + 1)
                ax[2].plot(x, y1, '--o', label="$" + name + "_{global}$")
                ax[2].plot(x, y2, '-o', label="$" + name + "_{local}$")
                ax[2].set(title='Num Examples (' + phase + ')', xlabel='round', ylabel='examples')
                ax[2].xaxis.set_ticks(x[0::1])
                ax[2].legend(loc=4)
                ax[2].grid()

                plt.tight_layout()
                plt.savefig(self.output_dir + "/round_configs_" + name + "_" + str(i) + "_" + phase + self.ext)
                if self.show_plot:
                    plt.show()
                plt.close()

    def plot_matrix_devices(self, phase="fit"):
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
                plt.savefig(self.output_dir + "/matrix_devices_" + name + "_" + str(j) + "_" + phase + self.ext)
                if self.show_plot:
                    plt.show()
                plt.close()

    def plot_devices_bar(self, phase="fit"):
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
                ax[r].legend(loc=1)
                ax[r].grid()

                plt.tight_layout()
                plt.savefig(self.output_dir + "/devs_bar_" + name + "_" + str(i) + "_" + phase + self.ext)
                if self.show_plot:
                    plt.show()
                plt.close()

    def plot_devices_capabilities_bar(self):
        for name, sim in self.sims:
            for i, status in enumerate(sim["status"]):
                fig, ax = plt.subplots(3)

                # availability
                r = 0
                y = status["con"]["devs"]["ips"]
                x = range(0, y.shape[0])
                ax[r].bar(x, y, label=name, alpha=0.5)
                ax[r].set(title='Computation Speed', xlabel='device', ylabel='IPS')
                ax[r].legend(loc=1)
                ax[r].grid()

                # energy
                r += 1
                val = status["con"]["devs"]["energy"]
                y = np.mean(val, axis=0)
                x = range(0, y.shape[0])
                ax[r].bar(x, y, label=name, alpha=0.5)
                ax[r].set(title='Available Energy', xlabel='device', ylabel='mAh')
                ax[r].legend(loc=1)
                ax[r].grid()

                # network
                r += 1
                val = status["con"]["devs"]["net_speed"]
                y = np.mean(val, axis=0)
                x = range(0, y.shape[0])
                ax[r].bar(x, y, label=name, alpha=0.5)
                ax[r].set(title='Network Speed', xlabel='device', ylabel='params/s')
                ax[r].legend(loc=1)
                ax[r].grid()

                plt.tight_layout()
                plt.savefig(self.output_dir + "/devs_capabilities_bar_" + name + "_" + str(i) + self.ext)
                if self.show_plot:
                    plt.show()
                plt.close()

    def plot_devs_data(self):
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
                plt.savefig(self.output_dir + "/devs_data_" + name + "_" + str(i) + self.ext)
                if self.show_plot:
                    plt.show()
                plt.close()

    def plot_devs_local_data(self):
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
                plt.savefig(self.output_dir + "/devs_local_data_" + name + "_" + str(i) + self.ext)
                if self.show_plot:
                    plt.show()
                plt.close()

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
