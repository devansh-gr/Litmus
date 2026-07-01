import AppKit
import ApplicationServices

/// Milestone 1 automatic capture path.
///
/// Each tick it resolves the *frontmost application* and reads `kAXSelectedText`
/// from that app's focused element. Resolving the frontmost app on every poll —
/// rather than reusing a single cached system-wide focused element — makes
/// selection tracking survive app switches and repeated selections (the earlier
/// version could get stuck on the first value it saw).
///
/// Native apps (TextEdit, Safari, Notes, Mail, Pages…) expose `kAXSelectedText`
/// directly. Chromium/Electron only expose their accessibility tree after the
/// `AXManualAccessibility` attribute is set, which we do below — but that path is
/// best-effort. For guaranteed coverage in Chrome/Electron/PDFs, use the
/// universal ⌘C hotkey (see `PasteboardCapture` / `HotkeyManager`).
final class SelectionTextCapture {

    private let interval: TimeInterval
    private let debug: Bool
    private let onSelection: (String) -> Void
    private let ownPID = ProcessInfo.processInfo.processIdentifier

    private var timer: DispatchSourceTimer?
    private var lastReported: String?
    private var lastDebugInfo: String?
    private var manualAccessibilityEnabled: Set<pid_t> = []

    init(interval: TimeInterval = 0.25, debug: Bool = true, onSelection: @escaping (String) -> Void) {
        self.interval = interval
        self.debug = debug
        self.onSelection = onSelection
    }

    func start() {
        let timer = DispatchSource.makeTimerSource(queue: .main)
        timer.schedule(deadline: .now(), repeating: interval)
        timer.setEventHandler { [weak self] in self?.poll() }
        timer.resume()
        self.timer = timer
    }

    func stop() {
        timer?.cancel()
        timer = nil
    }

    private func poll() {
        let result = readSelection()

        // Log only when the observed state *changes*, so the console shows
        // meaningful transitions instead of ~4 identical lines per second.
        if debug, result.info != lastDebugInfo {
            lastDebugInfo = result.info
            report("· \(result.info)")
        }

        guard let text = result.text, !text.isEmpty else { return }
        guard text != lastReported else { return }   // de-dupe unchanged selections
        lastReported = text
        onSelection(text)
    }

    /// Returns the current selection (if any) plus a short human-readable status
    /// describing what the accessibility read saw — used for debug logging.
    private func readSelection() -> (text: String?, info: String) {
        guard let app = NSWorkspace.shared.frontmostApplication else {
            return (nil, "no frontmost app")
        }
        let appName = app.localizedName ?? "pid \(app.processIdentifier)"
        if app.processIdentifier == ownPID {
            return (nil, "frontmost is self")
        }

        let appElement = AXUIElementCreateApplication(app.processIdentifier)
        enableManualAccessibilityIfNeeded(pid: app.processIdentifier, appElement: appElement)

        guard let focusedRef = copyAttribute(appElement, kAXFocusedUIElementAttribute),
              CFGetTypeID(focusedRef) == AXUIElementGetTypeID()
        else {
            return (nil, "\(appName): no focused element")
        }

        // Safe: confirmed above that this is an AXUIElement.
        let focused = focusedRef as! AXUIElement
        let role = (copyAttribute(focused, kAXRoleAttribute) as? String) ?? "?"

        if let selection = copyAttribute(focused, kAXSelectedTextAttribute) as? String,
           !selection.isEmpty {
            return (selection, "\(appName)/\(role) selectedLen=\(selection.count)")
        }
        return (nil, "\(appName)/\(role): kAXSelectedText empty/unavailable")
    }

    /// Chromium/Electron expose their a11y tree (and `kAXSelectedText` on web
    /// content) only after this attribute is set. Set once per process; harmless
    /// for apps that ignore it.
    private func enableManualAccessibilityIfNeeded(pid: pid_t, appElement: AXUIElement) {
        guard !manualAccessibilityEnabled.contains(pid) else { return }
        manualAccessibilityEnabled.insert(pid)
        AXUIElementSetAttributeValue(appElement, "AXManualAccessibility" as CFString, kCFBooleanTrue)
    }

    private func copyAttribute(_ element: AXUIElement, _ attribute: String) -> AnyObject? {
        var value: AnyObject?
        let result = AXUIElementCopyAttributeValue(element, attribute as CFString, &value)
        return result == .success ? value : nil
    }
}
