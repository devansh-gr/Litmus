import Foundation

/// Selects the classifier behind the `Classifier` protocol.
///
///   CPD_CLASSIFIER=mock    -> offline keyword heuristic (no server needed)
///   CPD_CLASSIFIER=remote  -> local inference server (default)
///   CPD_ENDPOINT_URL       -> override the server URL (default 127.0.0.1:8765)
enum Config {
    enum Backend: String {
        case mock
        case remote
    }

    static var backend: Backend {
        let raw = ProcessInfo.processInfo.environment["CPD_CLASSIFIER"]?.lowercased() ?? "remote"
        return Backend(rawValue: raw) ?? .remote
    }

    static func makeClassifier() -> Classifier {
        switch backend {
        case .mock:   return MockClassifier()
        case .remote: return RemoteClassifier()
        }
    }

    /// Only the remote backend can produce a brain map.
    static func makeBrainMapper() -> RemoteClassifier? {
        backend == .remote ? RemoteClassifier() : nil
    }

    /// Verdicts below this confidence are suppressed (no overlay).
    static var confidenceThreshold: Double {
        ProcessInfo.processInfo.environment["CPD_MIN_CONFIDENCE"]
            .flatMap(Double.init).map { $0 / 100.0 } ?? 0.30
    }
}
