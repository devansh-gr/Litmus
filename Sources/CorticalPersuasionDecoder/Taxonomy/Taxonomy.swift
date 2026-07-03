import Foundation

/// The persuasion vectors the decoder classifies. Raw values are the stable keys
/// a classifier (mock or remote) returns and looks up.
enum PersuasionVector: String, CaseIterable, Codable {
    case fearMongering               = "fear-mongering"
    case criticalThinkingSuppression = "critical-thinking-suppression"
    case tribalInGroupBias           = "tribal-in-group-bias"
    case dopamineBait                = "dopamine-bait"
    case outrage                     = "outrage"
    case authorityAppeal             = "authority-appeal"
    case falseUrgency                = "false-urgency"
    case socialProofConformity       = "social-proof-conformity"
}

/// Maps a vector to the brain region it targets and the cognitive mechanism.
/// Mechanisms cite the actual process (per project rule: neuroanatomy must be
/// defensible, no "lizard brain" hand-waving).
struct TaxonomyEntry {
    let vector: PersuasionVector
    let displayName: String
    let brainRegion: String
    let mechanism: String
}

enum Taxonomy {
    static let table: [PersuasionVector: TaxonomyEntry] = [
        .fearMongering: .init(
            vector: .fearMongering, displayName: "Fear-mongering",
            brainRegion: "Amygdala",
            mechanism: "Threat-salience detection drives rapid affective appraisal ahead of cortical evaluation."),
        .criticalThinkingSuppression: .init(
            vector: .criticalThinkingSuppression, displayName: "Critical-thinking suppression",
            brainRegion: "Dorsolateral prefrontal cortex",
            mechanism: "Working-memory / cognitive-load overload degrades deliberative reasoning."),
        .tribalInGroupBias: .init(
            vector: .tribalInGroupBias, displayName: "Tribal in-group bias",
            brainRegion: "Medial PFC + anterior insula",
            mechanism: "Self / in-group representation (mPFC) plus out-group affect (insula)."),
        .dopamineBait: .init(
            vector: .dopamineBait, displayName: "Dopamine bait",
            brainRegion: "Nucleus accumbens",
            mechanism: "Ventral-striatal reward-prediction and variable-ratio reinforcement."),
        .outrage: .init(
            vector: .outrage, displayName: "Outrage",
            brainRegion: "Anterior cingulate cortex",
            mechanism: "Conflict and affective-salience monitoring."),
        .authorityAppeal: .init(
            vector: .authorityAppeal, displayName: "Authority appeal",
            brainRegion: "Prefrontal cortex (deference)",
            mechanism: "Executive evaluation offloaded to a perceived authority."),
        .falseUrgency: .init(
            vector: .falseUrgency, displayName: "False urgency",
            brainRegion: "Amygdala + HPA axis",
            mechanism: "Acute stress response narrows the deliberation window under time pressure."),
        .socialProofConformity: .init(
            vector: .socialProofConformity, displayName: "Social-proof conformity",
            brainRegion: "Ventromedial prefrontal cortex",
            mechanism: "Value signal updated toward group consensus."),
    ]

    /// Table is exhaustive over the enum, so this is total.
    static func entry(for vector: PersuasionVector) -> TaxonomyEntry {
        table[vector]!
    }
}
