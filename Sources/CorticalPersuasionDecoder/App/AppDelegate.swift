import AppKit
import os

let log = Logger(subsystem: "ai.zeonsystems.corticalpersuasiondecoder", category: "capture")

/// Writes to both stderr (visible when run from Terminal / the Xcode console) and
/// the unified log (visible in Console.app, filtered by the subsystem above).
func report(_ message: String) {
    FileHandle.standardError.write(Data(("[CPD] " + message + "\n").utf8))
    log.log("\(message, privacy: .public)")
}

final class AppDelegate: NSObject, NSApplicationDelegate {

    private var capture: SelectionTextCapture?
    private var hotkey: HotkeyManager?
    private var permissionTimer: Timer?

    func applicationDidFinishLaunching(_ notification: Notification) {
        report("Cortical Persuasion Decoder — Milestone 1 (capture spike)")
        report("Auto: highlight text in native apps (TextEdit/Safari/Notes) — it prints automatically.")
        report("Universal: press ⌃⌥⌘C to capture the selection in ANY app (Chrome, Electron, PDFs).")
        ensureAccessibilityThenStart()
    }

    private func ensureAccessibilityThenStart() {
        if Permissions.isAccessibilityTrusted() {
            start()
            return
        }
        report("⚠️  Accessibility permission NOT granted yet.")
        report("    1. Approve the system dialog that just appeared.")
        report("    2. Enable this app under: System Settings ▸ Privacy & Security ▸ Accessibility")
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
            self?.start()
        }
    }

    private func start() {
        startAutoCapture()
        startHotkey()
    }

    private func startAutoCapture() {
        let capture = SelectionTextCapture { text in
            let preview = text.replacingOccurrences(of: "\n", with: " ⏎ ")
            report("📋 auto (\(text.count) chars): \(preview)")
        }
        capture.start()
        self.capture = capture
        report("👂 Auto capture running (polls the focused element every 250 ms).")
    }

    private func startHotkey() {
        let hotkey = HotkeyManager { [weak self] in self?.captureViaHotkey() }
        hotkey.register()
        self.hotkey = hotkey
        report("⌨️  Universal capture hotkey armed: ⌃⌥⌘C")
    }

    private func captureViaHotkey() {
        // Let the physical ⌃⌥ modifiers lift before synthesizing ⌘C, so they
        // don't merge into the copy keystroke.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.12) {
            guard let text = PasteboardCapture.selectedTextViaCopy(), !text.isEmpty else {
                report("⌨️  hotkey: nothing captured (is anything selected?)")
                return
            }
            let preview = text.replacingOccurrences(of: "\n", with: " ⏎ ")
            report("⌨️  captured (\(text.count) chars): \(preview)")
        }
    }
}
