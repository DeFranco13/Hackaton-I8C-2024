class Visits{
    id: number;
    rijksregisterPatient: string;
    rijksregisterNurse: string;
    rijksregiserScanType: ScanType;
    visitAmounts: number;
    duration: Duration;
    careType: HomeCareServices;
    distance: number; //in km
    driveTime: number; // in minutes
    nurseLocation: Location; // coordinates only in belgium
    patientLocation: Location; // coordinates only in belgium
    visit_timestamp: string; // mostly between 6 and 22h. What lays outside is ananomality so put some in the dataset
}

enum ScanType {
    MANUAL,
    EID,
    BARCODE,    
}

enum Duration {
    ONE_MONTH,
    TWO_MONTHS,
    // default
    THREE_MONTHS,
    FOUR_MONTHS,
    FIVE_MONTHS,
    SIX_MONTHS,
    SEVEN_MONTHS,
    EIGHT_MONTHS,
    NINE_MONTHS,
    TEN_MONTHS,
    ELEVEN_MONTHS,
    TWELVE_MONTHS
}


export class Location {
    latitude: number;
    longtitude: number;
}

enum HomeCareServices {
    ADMINISTER_MEDICATION,
    PILL_DISPENSING,
    INJECTION_ADMINISTRATION,
    INFUSION_THERAPY,

    PERSONAL_HYGIENE,
    BATHING,
    DENTAL_CARE,
    HAIR_CARE,

    NUTRITIONAL_SUPPORT,
    MEAL_PREPARATION,
    FEEDING_ASSISTANCE,
    DIET_MONITORING,

    MOBILITY_SUPPORT,
    PHYSICAL_THERAPY,
    WALKING_ASSISTANCE,
    TRANSFER_ASSISTANCE,

    VITAL_SIGN_MONITORING,
    BLOOD_PRESSURE_CHECK,
    TEMPERATURE_MONITORING,
    BLOOD_SUGAR_MONITORING
}
