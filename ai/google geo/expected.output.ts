class Output {
    type: Types;
    rijksregisterNurse: string;
    rijksregisterPatient: string;
    weight: number;
}

enum Types {
    GEO_LOCATION,
    VISIT_CHECK,
    CARE_MIX
}