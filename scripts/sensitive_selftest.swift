// Validates the SensitiveText secret-detection guard against the REAL source.
// Compile together with the actual implementation (no duplication):
//
//   swiftc scripts/sensitive_selftest.swift \
//          Sources/CorticalPersuasionDecoder/Support/SensitiveText.swift \
//          -o /tmp/sst && /tmp/sst
//
// Prints PASS if every case classifies as expected. This guards the safety feature
// that stops passwords / API keys / card numbers from being sent to the classifier.

import Foundation

@main
enum SensitiveSelfTest {
    // (secret?, text) — true = should be flagged as sensitive and skipped.
    static let cases: [(Bool, String)] = [
        // Secrets that MUST be caught:
        (true,  "sk-proj-abc123DEF456ghi789JKL012mno345"),
        (true,  "ghp_16CharsAtLeastxxxxxxxxxxxxxxxxxxxx"),
        (true,  "hf_abcdefghijklmnopqrstuvwxyz1234567890"),
        (true,  "my password is hunter2 do not share"),
        (true,  "4111 1111 1111 1111"),
        (true,  "eyJhbGciOi.eyJzdWIiOiIxMjM0NTY.SflKxwRJSMeKKF2QT4"),
        (true,  "AKIAIOSFODNN7EXAMPLE"),
        // Ordinary text that must NOT be flagged (would break normal ⌘B use):
        (false, "Act now or lose everything forever."),
        (false, "The meeting is scheduled for 3pm on Tuesday in room 214."),
        (false, "Everyone is switching, don't get left behind."),
        (false, "This will change your life and the entire world forever."),
        (false, "Call me at extension 214 when you arrive."),
    ]

    static func main() {
        var failures = 0
        for (expected, text) in cases {
            let got = SensitiveText.looksSensitive(text)
            let ok = got == expected
            if !ok { failures += 1 }
            let mark = ok ? "ok  " : "FAIL"
            let clip = text.count > 40 ? String(text.prefix(40)) + "…" : text
            print("\(mark)  expected=\(expected)  got=\(got)  \(clip)")
        }
        print(failures == 0
              ? "\nPASS \u{2713} — all \(cases.count) cases correct"
              : "\n\(failures) FAILURE(S) of \(cases.count)")
        exit(failures == 0 ? 0 : 1)
    }
}
