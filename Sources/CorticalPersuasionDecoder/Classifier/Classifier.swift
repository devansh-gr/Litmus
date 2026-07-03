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

/// A classification result. `confidence` is mandatory on every verdict (0.0–1.0)
/// — no unqualified calls, per project rule.
struct Verdict {
    let vector: PersuasionVector
    let confidence: Double
    let rationale: String?
}

/// The seam that decouples capture from classification. Implementations:
/// `MockClassifier` (offline, instant) and later `RemoteClassifier` (HTTP/JSON).
protocol Classifier {
    func classify(_ input: ClassificationInput) async throws -> Verdict
}
