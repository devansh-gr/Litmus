import Foundation

/// What gets sent to a classifier. Either or both fields may be set (selected
/// text and/or an image crop for OCR/vision).
struct ClassificationInput {
    let text: String?
    let imagePNG: Data?

    init(text: String? = nil, imagePNG: Data? = nil) {
        self.text = text
        self.imagePNG = imagePNG
    }
}

/// One vector with its probability — used for the runner-up mixture.
struct VectorScore {
    let vector: PersuasionVector
    let probability: Double
}

/// A classification result. `confidence` is mandatory on every verdict (0.0–1.0)
/// — no unqualified calls, per project rule. `alternatives` are the runner-up
/// vectors (real persuasion is a mixture, not a single label).
struct Verdict {
    let vector: PersuasionVector
    let confidence: Double
    let rationale: String?
    let alternatives: [VectorScore]
    /// Server flagged the confidence below its abstain floor: it's likely
    /// manipulative but the specific technique is a guess. The card says so.
    let uncertain: Bool

    init(vector: PersuasionVector, confidence: Double, rationale: String?,
         alternatives: [VectorScore] = [], uncertain: Bool = false) {
        self.vector = vector
        self.confidence = confidence
        self.rationale = rationale
        self.alternatives = alternatives
        self.uncertain = uncertain
    }
}

/// The seam that decouples capture from classification. Implementations:
/// `MockClassifier` (offline, instant) and later `RemoteClassifier` (HTTP/JSON).
protocol Classifier {
    func classify(_ input: ClassificationInput) async throws -> Verdict
}
