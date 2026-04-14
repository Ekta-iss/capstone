import numpy as np

def classify_cycle(future_position, velocity, risk_score):

    if velocity < 0.2 and risk_score < 0.3:
        return "IDLE"

    elif velocity > 0.2 and future_position < 0.4:
        return "LOADING"

    elif velocity > 0.5:
        return "MOVING"

    elif velocity < 0.2 and risk_score > 0.5:
        return "UNLOADING"

    else:
        return "TRANSITION"