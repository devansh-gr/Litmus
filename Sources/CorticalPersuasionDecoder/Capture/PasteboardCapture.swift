import AppKit
import CoreGraphics

/// Grabs the current selection by synthesizing ⌘C, reading the pasteboard, then
/// restoring the previous clipboard. Works in any app that supports Copy —
/// including Chrome/Safari article body and PDFs. Requires Accessibility
/// permission (to post the keystroke).
enum PasteboardCapture {

    static func selectedTextViaCopy(timeout: TimeInterval = 0.3) -> String? {
        let pasteboard = NSPasteboard.general
        let saved = snapshot(pasteboard)
        let changeCountBefore = pasteboard.changeCount

        postCommandC()

        // Wait briefly for the frontmost app to service ⌘C and update the pasteboard.
        let deadline = Date().addingTimeInterval(timeout)
        while pasteboard.changeCount == changeCountBefore && Date() < deadline {
            RunLoop.current.run(mode: .default, before: Date(timeIntervalSinceNow: 0.01))
        }

        let copied = pasteboard.string(forType: .string)
        restore(pasteboard, from: saved)
        return (copied?.isEmpty ?? true) ? nil : copied
    }

    private static func snapshot(_ pasteboard: NSPasteboard) -> [NSPasteboardItem] {
        pasteboard.pasteboardItems?.map { item in
            let copy = NSPasteboardItem()
            for type in item.types {
                if let data = item.data(forType: type) {
                    copy.setData(data, forType: type)
                }
            }
            return copy
        } ?? []
    }

    private static func restore(_ pasteboard: NSPasteboard, from items: [NSPasteboardItem]) {
        pasteboard.clearContents()
        if !items.isEmpty {
            pasteboard.writeObjects(items)
        }
    }

    private static func postCommandC() {
        let source = CGEventSource(stateID: .combinedSessionState)
        let cKey: CGKeyCode = 0x08  // kVK_ANSI_C
        let down = CGEvent(keyboardEventSource: source, virtualKey: cKey, keyDown: true)
        down?.flags = .maskCommand
        let up = CGEvent(keyboardEventSource: source, virtualKey: cKey, keyDown: false)
        up?.flags = .maskCommand
        down?.post(tap: .cgAnnotatedSessionEventTap)
        up?.post(tap: .cgAnnotatedSessionEventTap)
    }
}
