import logging
import coloredlogs
from fl_sim.analyzer import SimAnalyzer
import argparse

# init log
log_format = "%(message)s"
logging.basicConfig(level='DEBUG', format=log_format)
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger, fmt=log_format, datefmt="%H:%M:%S")

parser = argparse.ArgumentParser()
parser.add_argument("-f", nargs='+', dest='files', type=str, required=True)
args = parser.parse_args()

fl = [(file[:file.rfind('.')], file) for file in args.files]
logger.info("reading: {}".format(fl))

analyzer = SimAnalyzer(fl, "output", "graphs", logger, extension="png", show_plot=False, show_data=False)

analyzer.plot_agg_accuracy_loss(phase="eval")
analyzer.plot_round_consumptions(phase="fit")
analyzer.plot_round_times(phase="fit")
analyzer.plot_round_configs(phase="fit")
analyzer.plot_round_devices(phase="fit")
analyzer.plot_devs_data()
analyzer.plot_matrix_devices(phase="fit")
analyzer.plot_devices_bar(phase="fit")
analyzer.plot_devices_capabilities_bar()
analyzer.print_metrics()
# analyzer.print_data()
