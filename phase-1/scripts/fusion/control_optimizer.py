def optimize_control(lstm_pred, risk_score, cycle_time=None):

    future_dist, angle, risk = lstm_pred

    speed = 1.0
    action = "NORMAL"
    reason = "Safe operation"

    # risk-based control
    if risk_score > 0.7:
        action = "STOP"
        speed = 0.2
        reason = "High risk detected"
    elif risk_score > 0.4:
        action = "SLOW"
        speed = 0.5
        reason = "Moderate risk detected"

    # cycle time adjustment (NEW LOGIC)
    if cycle_time is not None:
        if cycle_time > 30:
            speed *= 0.8
            reason += " | High cycle time reducing speed"

    return {
        "action": action,
        "speed": round(speed, 2),
        "reason": reason,
        "cycle_time": cycle_time
    }