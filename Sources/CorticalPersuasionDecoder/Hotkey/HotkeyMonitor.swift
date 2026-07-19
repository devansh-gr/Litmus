import AppKit

/// A global hotkey via an `NSEvent` monitor (requires Accessibility).
///
/// Deliberately **non-consuming**: unlike a Carbon `RegisterEventHotKey`, this
/// observes the keystroke without swallowing it, so the same combo still works in
/// the focused app (e.g. ⌘B still bolds in an editor) while also triggering us.
/// This tool is used on *reading* surfaces (X, articles, search) where ⌘B is a
/// no-op natively, so it just becomes "analyze this selection."
///
/// Default: ⌘B. Override the key with the CPD_HOTKEY env var (a single letter).
final class HotkeyMonitor {

    private var monitor: Any?
    private let keyCode: UInt16
    private let modifiers: NSEvent.ModifierFlags
    private let onTrigger: () -> Void

    init(onTrigger: @escaping () -> Void) {
        self.onTrigger = onTrigger
        self.modifiers = [.command]
        // 'B' = 0x0B. Allow a one-letter override (e.g. CPD_HOTKEY=D).
        if let letter = ProcessInfo.processInfo.environment["CPD_HOTKEY"]?.uppercased(),
           let code = Self.keyCodes[letter] {
            self.keyCode = code
        } else {
            self.keyCode = 0x0B
        }
    }

    func start() {
        monitor = NSEvent.addGlobalMonitorForEvents(matching: .keyDown) { [weak self] event in
            guard let self else { return }
            let mods = event.modifierFlags.intersection([.command, .option, .control, .shift])
            if event.keyCode == self.keyCode && mods == self.modifiers {
                self.onTrigger()
            }
        }
    }

    func stop() {
        if let monitor { NSEvent.removeMonitor(monitor); self.monitor = nil }
    }

    deinit { stop() }

    /// Virtual key codes for the letter keys we allow as an override.
    private static let keyCodes: [String: UInt16] = [
        "A": 0x00, "B": 0x0B, "C": 0x08, "D": 0x02, "E": 0x0E, "F": 0x03,
        "G": 0x05, "M": 0x2E, "P": 0x23, "R": 0x0F, "S": 0x01, "V": 0x09,
    ]
}
