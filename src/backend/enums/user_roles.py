from enum import Enum


class UserRole(str, Enum):
    CLINICIAN = "clinician"
    DEVELOPER = "developer"
