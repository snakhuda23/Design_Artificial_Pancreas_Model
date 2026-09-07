import numpy as np

class PIDcontroller:
    """
    Tuned PID Controller for Fast Intraperitoneal (IP) Insulin Delivery
    """
    def __init__(
        self,
        target_glucose: float = 110.0,    # mg/dL (Basal target)
        r_basal: float = 59.6976,         # mU/min (Patient fasting basal rate)
        Kp: float = 0.8077, #1.25,                 # Proportional gain (mU/min per mg/dL)
        Ti: float = 273.0, #350.0,                # Integral time constant (min)
        Td: float =  23.5, #15.0,                 # Derivative time constant (min)
        u_max: float = 15000.0,             # was 300; Maximum pump infusion cap (mU/min)
        sample_time_min: float = 1.0,     # dt (min)
    ):
        self.target = target_glucose
        self.r_basal = r_basal
        self.dt = sample_time_min
        self.Kp = Kp
        self.Ti = Ti
        self.Td = Td
        self.u_max = u_max

        # Internal controller state
        self.integral = 0.0
        self.prev_glucose = target_glucose

    def compute(self, SG_n: float) -> float:
        """
        Computes the required IP insulin infusion rate (mU/min).
        """
        error = SG_n - self.target
 
        # 1. Proportional Term
        P = self.Kp * error

        # 2. Derivative Term on measurement (rate of change)
        dSGdt = (SG_n - self.prev_glucose) / self.dt
        D = self.Kp * self.Td * dSGdt

        # 3. Integral Term with Anti-Windup
        tentative_integral = self.integral + error * self.dt
        I = (self.Kp / self.Ti) * tentative_integral

        # 4. Total unconstrained command (with basal feedforward)
        u_raw = self.r_basal + P + I + D

        # 5. Apply Pump Physical Bounds [0, u_max] and Anti-Windup Clamping
        if u_raw < 0.0:
            u_clamped = 0.0
            if error > 0:
                self.integral = tentative_integral
        elif u_raw > self.u_max:
            u_clamped = self.u_max
            if error < 0:
                self.integral = tentative_integral
        else:
            u_clamped = u_raw
            self.integral = tentative_integral

        self.prev_glucose = SG_n

        return float(u_clamped)