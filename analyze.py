import logging
import coloredlogs
from fl_anlz.sim_analyzer import SimAnalyzer
import argparse

# init log
log_format = "%(message)s"
logging.basicConfig(level='DEBUG', format=log_format)
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger, fmt=log_format, datefmt="%H:%M:%S")

parser = argparse.ArgumentParser()
parser.add_argument("-f", nargs='+', dest='files', type=str, help="Input files", required=True)
parser.add_argument("-p", dest="do_plot", help="Plot graphs", action='store_true')
parser.add_argument("-d", dest="show_data", help="Show simulation data", action='store_true')
args = parser.parse_args()

fl = [(file[:file.rfind('.')], file) for file in args.files]
logger.info("reading: {}".format(fl))

analyzer = SimAnalyzer(fl, "output", "graphs", logger, extension="png", show_plot=False, show_data=False)

if args.do_plot:
    analyzer.plot_accuracy(phase="eval", color="k")
    analyzer.plot_loss(phase="eval", color="k")
    analyzer.plot_computation_time(phase="fit", color="k")
    analyzer.plot_communication_time(phase="fit", color="k")
    analyzer.plot_total_time(phase="fit", color="k")
    analyzer.plot_resources_consumption(phase="fit", color="k")
    analyzer.plot_network_consumption(phase="fit", color="k")
    analyzer.plot_energy_consumption(phase="fit", color="k")
    analyzer.plot_available_devices(color="k")
    analyzer.plot_selected_devices(phase="fit", color="k")
    analyzer.plot_available_failed_devices(color="k")
    analyzer.plot_selected_successful_devices(phase="fit", color="k")
    analyzer.plot_epochs_config(phase="fit", color="k")
    analyzer.plot_batch_size_config(phase="fit", color="k")
    analyzer.plot_num_examples_config(phase="fit", color="k")
    analyzer.plot_matrix_devices(phase="fit")
    analyzer.plot_devices_bar_availability(phase="fit", color="k")
    analyzer.plot_devices_bar_failures(phase="fit", color="k")
    analyzer.plot_devices_bar_selected(phase="fit", color="k")
    analyzer.plot_devices_data_size()
    analyzer.plot_devices_ips()
    analyzer.plot_devices_available_energy()
    analyzer.plot_devices_network_speed()
    analyzer.plot_devices_data_distribution()

if args.show_data:
    analyzer.print_availability()
    analyzer.print_failures()
    analyzer.print_ips()
    analyzer.print_energy()
    analyzer.print_net_speed()
    analyzer.print_local_data_size()
    analyzer.print_model_params()
    analyzer.print_selection(phase="fit")
    analyzer.print_selection(phase="eval")
    analyzer.print_total_time(phase="fit")
    analyzer.print_total_time(phase="eval")
    analyzer.print_resources_consumption(phase="fit")
    analyzer.print_resources_consumption(phase="eval")
    analyzer.print_energy_consumption(phase="fit")
    analyzer.print_energy_consumption(phase="eval")
    analyzer.print_network_consumption(phase="fit")
    analyzer.print_network_consumption(phase="eval")
    analyzer.print_accuracy(phase="fit")
    analyzer.print_accuracy(phase="eval")
    analyzer.print_loss(phase="fit")
    analyzer.print_loss(phase="eval")

analyzer.close()
