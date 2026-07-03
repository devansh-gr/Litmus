import AppKit

/// Universal capture path (Milestone 1).
///
/// Watches the general pasteboard and reports newly-copied text. The user selects
/// text in ANY app — Chrome article body, a PDF, a native app — and presses ⌘C;
/// we read what they copied. No global hotkey, no synthesized keystrokes, no
/// per-app accessibility quirks, and — crucially — **no special permissions**
/// (reading the pasteboard is unrestricted). This is the reliable demo path for
/// web/article content, which the Accessibility API cannot read directly.
final class PasteboardWatcher {

    private let pasteboard = NSPasteboard.general
    private let interval: TimeInterval
    private let onCopy: (String) -> Void
    private var timer: DispatchSourceTimer?
    private var lastChangeCount: Int
    private var lastText: String?

    init(interval: TimeInterval = 0.2, onCopy: @escaping (String) -> Void) {
        self.interval = interval
        self.onCopy = onCopy
        // Start from the current change count so we don't fire on whatever was
        // already on the clipboard at launch.
        self.lastChangeCount = NSPasteboard.general.changeCount
    }

    func start() {
        let timer = DispatchSource.makeTimerSource(queue: .main)
        timer.schedule(deadline: .now() + interval, repeating: interval)
        timer.setEventHandler { [weak self] in self?.poll() }
        timer.resume()
        self.timer = timer
    }

    func stop() {
        timer?.cancel()
        timer = nil
    }

    private func poll() {
        let count = pasteboard.changeCount
        guard count != lastChangeCount else { return }
        lastChangeCount = count

        guard let text = pasteboard.string(forType: .string), !text.isEmpty else { return }
        // Chrome/others may write the pasteboard in several passes per ⌘C
        // (bumping changeCount each time); collapse identical repeats.
        guard text != lastText else { return }
        lastText = text
        onCopy(text)
    }
}
