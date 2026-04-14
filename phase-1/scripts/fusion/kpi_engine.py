class KPIEngine:

    def __init__(self):
        self.idle_time = 0
        self.throughput = 0
        self.safety_events = 0

    def update(self, cycle_state, risk):

        if cycle_state == "IDLE":
            self.idle_time += 1

        if cycle_state == "COMPLETE":
            self.throughput += 1

        if risk > 0.7:
            self.safety_events += 1

    def report(self):

        return {
            "idle_time": self.idle_time,
            "throughput": self.throughput,
            "safety_events": self.safety_events
        }