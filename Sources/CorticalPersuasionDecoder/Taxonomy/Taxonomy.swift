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
    case hypeHopeMongering           = "hype-hope-mongering"
    case fomo                        = "fomo"
    case manufacturedAwe             = "manufactured-awe"
    case guiltTripping               = "guilt-tripping"
    case loveBombing                 = "love-bombing"
}

/// Display name + the cognitive *technique* a vector uses.
///
/// NOTE: this deliberately makes NO per-vector brain-region claim. An earlier
/// version asserted fear→amygdala, dopamine→nucleus accumbens, etc., but our own
/// experiments (A3) showed those vectors do NOT separate by region, and the model
/// is cortex-only and can't see the amygdala. Any anatomy shown to the user comes
/// only from the *measured* cortical impact profile (`/brainmap`), never from here.
struct TaxonomyEntry {
    let vector: PersuasionVector
    let displayName: String
    /// How the technique acts on your thinking — a psychological claim, not anatomy.
    let mechanism: String
}

enum Taxonomy {
    static let table: [PersuasionVector: TaxonomyEntry] = [
        .fearMongering: .init(
            vector: .fearMongering, displayName: "Fear-mongering",
            mechanism: "Substitutes a vivid threat for reasoned appraisal, pushing you to react before you evaluate."),
        .criticalThinkingSuppression: .init(
            vector: .criticalThinkingSuppression, displayName: "Critical-thinking suppression",
            mechanism: "Discourages questioning or checking, so a claim is accepted without scrutiny."),
        .tribalInGroupBias: .init(
            vector: .tribalInGroupBias, displayName: "Tribal in-group bias",
            mechanism: "Frames it as us-versus-them, so group loyalty overrides the argument's merits."),
        .dopamineBait: .init(
            vector: .dopamineBait, displayName: "Dopamine bait",
            mechanism: "Dangles a reward or novelty to hijack attention toward the promised payoff."),
        .outrage: .init(
            vector: .outrage, displayName: "Outrage",
            mechanism: "Provokes moral anger at a target, which crowds out measured judgment and drives sharing."),
        .authorityAppeal: .init(
            vector: .authorityAppeal, displayName: "Authority appeal",
            mechanism: "Leans on experts or officials so their status substitutes for evidence you could check."),
        .falseUrgency: .init(
            vector: .falseUrgency, displayName: "False urgency",
            mechanism: "Imposes an artificial deadline so time pressure short-circuits deliberation."),
        .socialProofConformity: .init(
            vector: .socialProofConformity, displayName: "Social-proof conformity",
            mechanism: "Signals that everyone is doing it, so fear of being left out replaces independent judgment."),
        .hypeHopeMongering: .init(
            vector: .hypeHopeMongering, displayName: "Hype / hope-mongering",
            mechanism: "Inflates an exciting or utopian upside so desire and optimism outrun scrutiny of the claim."),
        .fomo: .init(
            vector: .fomo, displayName: "FOMO",
            mechanism: "Warns you'll be left behind if you don't act, so anxiety about missing out drives adoption."),
        .manufacturedAwe: .init(
            vector: .manufacturedAwe, displayName: "Manufactured awe",
            mechanism: "Frames it as revolutionary and unprecedented, so awe overwhelms your skepticism."),
        .guiltTripping: .init(
            vector: .guiltTripping, displayName: "Guilt-tripping",
            mechanism: "Uses guilt or obligation as leverage — 'if you really cared, you would' — so you comply to relieve the guilt, not because it's right."),
        .loveBombing: .init(
            vector: .loveBombing, displayName: "Love-bombing",
            mechanism: "Overwhelms you with flattery, affection, or grand promises to disarm you and build dependency, so devotion replaces judgment."),
    ]

    /// Table is exhaustive over the enum, so this is total.
    static func entry(for vector: PersuasionVector) -> TaxonomyEntry {
        table[vector]!
    }
}
