def optimize_control(lstm_pred, risk_score):

    """
    Returns crane control decision
    """

    future_dist, angle, predicted_risk = lstm_pred

    # SAFE STATE
    if risk_score > 0.7 or predicted_risk > 0.7:
        return {
            "action": "STOP",
            "speed": 0,
            "reason": "HIGH_RISK"
        }

    # CAUTIOUS STATE
    if risk_score > 0.4:
        return {
            "action": "SLOW",
            "speed": 0.3,
            "reason": "MEDIUM_RISK"
        }

    # OPTIMAL STATE
    if future_dist < 5:
        return {
            "action": "ALIGN_AND_PICK",
            "speed": 0.7,
            "reason": "OBJECT_IN_RANGE"
        }

    # IDLE MOVEMENT
    return {
        "action": "SEARCH",
        "speed": 0.5,
        "reason": "NO_TARGET"
    }