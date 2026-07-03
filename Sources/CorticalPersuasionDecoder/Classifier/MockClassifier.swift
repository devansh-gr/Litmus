import Foundation

/// Offline, instant, deterministic stand-in for a real model. Scores text against
/// keyword cues per vector so the whole capture→classify→label UX can be built and
/// demoed with no network. This is NOT a real classifier — its confidence is a
/// crude heuristic — but it exercises the exact `Classifier` seam the remote model
/// will drop into later.
struct MockClassifier: Classifier {

    private static let cues: [PersuasionVector: [String]] = [
        .fearMongering: ["danger", "threat", "terrifying", "catastrophe", "warning", "deadly",
                         "crisis", "collapse", "destroy", "fear", "panic", "disaster"],
        .criticalThinkingSuppression: ["obviously", "everyone knows", "don't overthink", "just trust",
                         "no need to", "common sense", "wake up", "do your own research", "simple as that"],
        .tribalInGroupBias: ["us vs them", "they want", "our people", "those people", "real ones",
                         "the elite", "outsiders", "true patriots", "the enemy", "globalists"],
        .dopamineBait: ["you won't believe", "shocking", "amazing", "free", "instant", "unlock",
                         "exclusive", "limited drop", "win", "jackpot", "insane"],
        .outrage: ["outrageous", "disgusting", "how dare", "furious", "enraged", "unacceptable",
                         "scandal", "betrayal", "shameful", "sickening"],
        .authorityAppeal: ["experts say", "scientists agree", "doctors recommend", "officials",
                         "authorities", "studies show", "according to experts", "the government confirms"],
        .falseUrgency: ["act now", "hurry", "last chance", "limited time", "expires", "only today",
                         "don't wait", "before it's too late", "ending soon", "act fast"],
        .socialProofConformity: ["everyone is", "join millions", "most people", "trending", "going viral",
                         "others like you", "thousands already", "don't be left out", "join the movement"],
    ]

    func classify(_ input: ClassificationInput) async throws -> Verdict {
        let text = (input.text ?? "").lowercased()
        guard !text.isEmpty else {
            return Verdict(vector: .criticalThinkingSuppression, confidence: 0.05,
                           rationale: "No text to analyze.")
        }

        var bestVector: PersuasionVector = .dopamineBait
        var bestHits: [String] = []
        for vector in PersuasionVector.allCases {
            let hits = (Self.cues[vector] ?? []).filter { text.contains($0) }
            if hits.count > bestHits.count {
                bestHits = hits
                bestVector = vector
            }
        }

        guard !bestHits.isEmpty else {
            return Verdict(vector: .dopamineBait, confidence: 0.12,
                           rationale: "No strong cues detected — low-confidence default.")
        }

        // Heuristic confidence scales with the number of distinct cue hits.
        let confidence = min(0.95, 0.40 + Double(bestHits.count) * 0.15)
        let rationale = "Matched cues: " + bestHits.joined(separator: ", ")
        return Verdict(vector: bestVector, confidence: confidence, rationale: rationale)
    }
}
