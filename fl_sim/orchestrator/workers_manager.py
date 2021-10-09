class WorkersManager:

    def __init__(self):
        self.workers_ip_addresses = []
        self.workers_port_numbers = []

    def add_worker(self, address, port):
        self.workers_ip_addresses.append(address)
        self.workers_port_numbers.append(port)

    def get_worker_addresses(self):
        return self.workers_ip_addresses

    def get_worker_ports(self):
        return self.workers_port_numbers
