import AppKit
import os

let log = Logger(subsystem: "ai.zeonsystems.corticalpersuasiondecoder", category: "capture")

/// Writes to both stderr (visible when run from Terminal) and the unified log
/// (visible in Console.app, filtered by the subsystem above).
func report(_ message: String) {
    FileHandle.standardError.write(Data(("[CPD] " + message + "\n").utf8))
    log.log("\(message, privacy: .public)")
}

final class AppDelegate: NSObject, NSApplicationDelegate {

    private var capture: SelectionTextCapture?
    private var permissionTimer: Timer?

    func applicationDidFinishLaunching(_ notification: Notification) {
        report("Cortical Persuasion Decoder — Milestone 1 (capture spike)")
        report("Highlight text in TextEdit / Safari / Notes; selections print below.")
        ensureAccessibilityThenStart()
    }

    private func ensureAccessibilityThenStart() {
        if Permissions.isAccessibilityTrusted() {
            startCapture()
            return
        }

        report("⚠️  Accessibility permission NOT granted yet.")
        report("    1. Approve the system dialog that just appeared.")
        report("    2. Enable this app under:")
        report("       System Settings ▸ Privacy & Security ▸ Accessibility")
        report("    Capture starts automatically once granted…")

        Permissions.promptForAccessibility()
        Permissions.openAccessibilitySettings()
        waitForAccessibility()
    }

    private func waitForAccessibility() {
        permissionTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] timer in
            guard Permissions.isAccessibilityTrusted() else { return }
            timer.invalidate()
            self?.permissionTimer = nil
            report("✅ Accessibility granted.")
            self?.startCapture()
        }
    }

    private func startCapture() {
        let capture = SelectionTextCapture { text in
            let preview = text.replacingOccurrences(of: "\n", with: " ⏎ ")
            report("📋 selected (\(text.count) chars): \(preview)")
        }
        capture.start()
        self.capture = capture
        report("👂 Capture running — polling the focused element every 300 ms.")
    }
}
