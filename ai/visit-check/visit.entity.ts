class Main {
    rijksregisterPatient: string; // example: "13975100197"
    rijksregisterNurse: string; // example: "13975100197"
    visit: Visits;
    visit_timestamp: string;
}
export class Location {
    latitude: number;
    longtitude: number;
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

class Visits{
    id: number;
    visitAmounts: number;
    duration: Duration;
    service: HomeCareServices
    nurseLocation: Location; // coordinates only in belgium
    patientLocation: Location; // coordinates only in belgium
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