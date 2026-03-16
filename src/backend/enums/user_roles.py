from enum import Enum


class UserRole(str, Enum):
    CLINICIAN = "clinician"
    DEVELOPER = "developer"


class ModelVisibility(str, Enum):
    PRIVATE = "private"
    CLINICIAN = "clinician"
    COMMUNITY = "community"
    CLINICIAN_AND_COMMUNITY = "clinician_and_community"
