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

    /// Primary Milestone 1 capture path: select text anywhere + ⌘C.
    private var pasteboardWatcher: PasteboardWatcher?

    // NOTE: `SelectionTextCapture` (Accessibility polling) and `HotkeyManager`
    // (Carbon global hotkey) still exist in the project but are intentionally
    // NOT started here. The AX path can't read web/article body text (it only
    // sees editable fields), and the ⌥⌘-style hotkeys collided with Chrome's
    // DevTools shortcut. The pasteboard watcher below is the reliable universal
    // path. Those files will be reused in later milestones.

    func applicationDidFinishLaunching(_ notification: Notification) {
        report("Cortical Persuasion Decoder — Milestone 1 (capture spike)")
        report("Interaction: select text in ANY app (Chrome article, PDF, native) and press ⌘C.")
        report("It reads what you copied — no special permissions, no hotkey conflicts.")
        startPasteboardWatch()
    }

    private func startPasteboardWatch() {
        let watcher = PasteboardWatcher { text in
            let oneLine = text.replacingOccurrences(of: "\n", with: " ⏎ ")
            let clipped = oneLine.count > 140 ? String(oneLine.prefix(140)) + "…" : oneLine
            report("✂️  copied (\(text.count) chars): \(clipped)")
        }
        watcher.start()
        pasteboardWatcher = watcher
        report("✂️  Ready — select any text and press ⌘C to capture it.")
    }
}
