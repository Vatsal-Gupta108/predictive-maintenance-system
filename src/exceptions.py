class PredictiveMaintenanceException(Exception):
    """Base exception class for all custom errors in the Predictive Maintenance system."""
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code

class DataValidationError(PredictiveMaintenanceException):
    """Raised when telemetry or input sensors violate physical boundaries."""
    def __init__(self, message: str):
        super().__init__(message, status_code=400)

class ModelNotLoadedError(PredictiveMaintenanceException):
    """Raised when model weights or artifacts are unavailable during prediction requests."""
    def __init__(self, message: str = "ML model artifacts are not loaded on the server."):
        super().__init__(message, status_code=503)

class InferenceError(PredictiveMaintenanceException):
    """Raised when numerical predictions or feature calculations fail during execution."""
    def __init__(self, message: str):
        super().__init__(message, status_code=500)
