import Foundation

/// A brain region measured by TRIBE v2, with its content-driven activation
/// (z-scored against a baseline corpus, so it is NOT just "there is speech").
struct BrainRegion: Decodable {
    let region: String
    let activationZ: Double

    enum CodingKeys: String, CodingKey {
        case region
        case activationZ = "activation_z"
    }
}

/// Vendor-neutral HTTP/JSON classifier. Points at the local inference server by
/// default (nothing leaves the machine); override with CPD_ENDPOINT_URL.
///
/// The split is deliberate and evidence-based (see the vault write-up):
///   /classify  — the LLM names the persuasion vector. FAST (~3s). It is the
///                better DETECTOR: a Llama embedding scored 100% vs the brain
///                map's 75%, and information theory says the brain map (a
///                function of the text) can never beat the text.
///   /brainmap  — TRIBE v2 says WHERE in cortex the content lands. SLOW (~30s+).
///                It is the INTERPRETER, never the detector.
struct RemoteClassifier: Classifier {

    let endpoint: URL
    let timeout: TimeInterval

    init(endpoint: URL? = nil, timeout: TimeInterval = 20) {
        let fromEnv = ProcessInfo.processInfo.environment["CPD_ENDPOINT_URL"]
            .flatMap(URL.init(string:))
        self.endpoint = endpoint ?? fromEnv ?? URL(string: "http://127.0.0.1:8765")!
        self.timeout = timeout
    }

    private struct ClassifyResponse: Decodable {
        let vector: String
        let confidence: Int
        let rationale: String?
    }

    private struct BrainMapResponse: Decodable {
        let topRegions: [BrainRegion]

        enum CodingKeys: String, CodingKey {
            case topRegions = "top_regions"
        }
    }

    // MARK: - Detection (fast)

    func classify(_ input: ClassificationInput) async throws -> Verdict {
        guard let text = input.text, !text.isEmpty else {
            throw RemoteClassifierError.noText
        }
        let response: ClassifyResponse = try await post(path: "/classify", text: text)

        guard let vector = PersuasionVector(rawValue: response.vector) else {
            // The server answers "none" for neutral content, which has no vector.
            throw RemoteClassifierError.neutral
        }
        return Verdict(
            vector: vector,
            confidence: Double(response.confidence) / 100.0,
            rationale: response.rationale
        )
    }

    // MARK: - Interpretation (slow — call this off the interactive path)

    func brainMap(for text: String) async throws -> [BrainRegion] {
        let response: BrainMapResponse = try await post(
            path: "/brainmap", text: text, timeout: 300
        )
        return response.topRegions
    }

    // MARK: - Transport

    private func post<T: Decodable>(
        path: String, text: String, timeout: TimeInterval? = nil
    ) async throws -> T {
        var request = URLRequest(url: endpoint.appendingPathComponent(path))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = timeout ?? self.timeout
        request.httpBody = try JSONSerialization.data(withJSONObject: ["text": text])

        if let key = ProcessInfo.processInfo.environment["CPD_API_KEY"], !key.isEmpty {
            request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        }

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw RemoteClassifierError.badStatus(
                (response as? HTTPURLResponse)?.statusCode ?? -1
            )
        }
        return try JSONDecoder().decode(T.self, from: data)
    }
}

enum RemoteClassifierError: Error, LocalizedError {
    case noText
    case neutral
    case badStatus(Int)

    var errorDescription: String? {
        switch self {
        case .noText: return "No text to classify."
        case .neutral: return "Classified as neutral — no persuasion vector."
        case .badStatus(let code): return "Inference server returned HTTP \(code)."
        }
    }
}
