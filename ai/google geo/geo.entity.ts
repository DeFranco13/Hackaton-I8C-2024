class GoogleGeo {
    rijksregisterPatient: string;
    rijksregisterNurse: string;
    // location only be in Belgium
    nurseLocation: Location; // coordinates
    patientLocation: Location; // coordinates
    distance: number;
    driveTime: number; // in minutes
    careType: HomeCareServices;
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
