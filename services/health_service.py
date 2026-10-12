from config import HEALTH_WARNING_THRESHOLD, HEALTH_CRITICAL_THRESHOLD


class HealthService:

    def get_health_status(self, failure_probability):

        if failure_probability >= HEALTH_CRITICAL_THRESHOLD:
            return "Critical"

        elif failure_probability >= HEALTH_WARNING_THRESHOLD:
            return "Warning"

        else:
            return "Healthy"