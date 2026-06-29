import ApplicationServices
import Foundation

/// Milestone 1 capture spike.
///
/// Polls the system-wide focused UI element for its selected text and reports
/// each new selection. This is the AX-only path — it works in apps that expose
/// `kAXSelectedText` (TextEdit, Safari, Notes, Mail, Pages, etc.). Apps that do
/// not expose it (notably Chrome / Electron web content) return nothing here;
/// the ⌘C + NSPasteboard fallback for those arrives in a later milestone.
final class SelectionTextCapture {

    private let systemWide = AXUIElementCreateSystemWide()
    private let interval: TimeInterval
    private let onSelection: (String) -> Void
    private var timer: DispatchSourceTimer?
    private var lastReported: String?

    init(interval: TimeInterval = 0.3, onSelection: @escaping (String) -> Void) {
        self.interval = interval
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
        guard let text = currentSelectedText(), !text.isEmpty else { return }
        guard text != lastReported else { return }   // de-dupe unchanged selections
        lastReported = text
        onSelection(text)
    }

    /// Reads `kAXSelectedText` from the currently focused element, system-wide.
    func currentSelectedText() -> String? {
        guard let focusedRef = copyAttribute(systemWide, kAXFocusedUIElementAttribute),
              CFGetTypeID(focusedRef) == AXUIElementGetTypeID()
        else { return nil }

        // Safe: confirmed above that this is an AXUIElement.
        let focused = focusedRef as! AXUIElement
        return copyAttribute(focused, kAXSelectedTextAttribute) as? String
    }

    private func copyAttribute(_ element: AXUIElement, _ attribute: String) -> AnyObject? {
        var value: AnyObject?
        let result = AXUIElementCopyAttributeValue(element, attribute as CFString, &value)
        return result == .success ? value : nil
    }
}
