import Foundation

/// Heuristic guard so we don't route copied passwords / API keys / card numbers
/// through the classifier. Everything is local, but analysing a secret is still a
/// liability — and pointless. Tuned to almost never fire on ordinary prose (which
/// contains whitespace) and to catch obvious credential shapes.
enum SensitiveText {
    static func looksSensitive(_ text: String) -> Bool {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty else { return false }
        let lower = t.lowercased()

        // Explicit secret markers / known key prefixes.
        if lower.contains("password") || lower.contains("passwd") || lower.contains("secret key") {
            return true
        }
        let prefixes = ["sk-", "hf_", "ghp_", "github_pat_", "xox", "aws_", "akia", "-----begin"]
        if prefixes.contains(where: { lower.hasPrefix($0) }) { return true }

        // JWT: three base64url segments separated by dots.
        let parts = t.split(separator: ".")
        if parts.count == 3, parts.allSatisfy({ $0.count >= 8 && isTokenChars($0) }) { return true }

        // Card / account numbers — mostly digits once spaces & dashes are stripped.
        let compact = t.filter { !$0.isWhitespace && $0 != "-" }
        let compactDigits = compact.filter { $0.isNumber }.count
        if compactDigits >= 13 && Double(compactDigits) / Double(max(compact.count, 1)) > 0.9 {
            return true
        }

        // A single token (no whitespace) is where credentials live; prose has spaces.
        if !t.contains(where: { $0.isWhitespace }) {
            let hasLetter = t.contains { $0.isLetter }
            let hasDigit = t.contains { $0.isNumber }
            if t.count >= 20 && hasLetter && hasDigit { return true }   // API-key-shaped
            let digits = t.filter { $0.isNumber }.count
            if digits >= 13 && Double(digits) / Double(t.count) > 0.7 { return true }  // card / account #
        }
        return false
    }

    private static func isTokenChars(_ s: Substring) -> Bool {
        s.allSatisfy { $0.isLetter || $0.isNumber || "-_=+/".contains($0) }
    }
}
