import AppKit

/// Presents a translucent full-screen overlay and lets the user drag a rectangle.
/// Calls back with the selected rect in global AppKit screen coordinates
/// (bottom-left origin, points), or nil if cancelled (Esc / zero-size drag).
///
/// Milestone 2 covers the main display only; multi-display drag is a later refinement.
final class RegionSelector {
    private var window: SelectionWindow?

    func begin(_ completion: @escaping (CGRect?) -> Void) {
        let frame = NSScreen.main?.frame ?? .zero
        let window = SelectionWindow(screenFrame: frame)
        window.onFinish = { [weak self] rect in
            window.orderOut(nil)
            self?.window = nil
            completion(rect)
        }
        self.window = window
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
}

final class SelectionWindow: NSWindow {
    var onFinish: ((CGRect?) -> Void)?
    private var startPoint: NSPoint?
    private let selectionView = SelectionView()

    init(screenFrame: NSRect) {
        super.init(contentRect: screenFrame, styleMask: .borderless, backing: .buffered, defer: false)
        isOpaque = false
        backgroundColor = NSColor.black.withAlphaComponent(0.15)
        level = .screenSaver
        ignoresMouseEvents = false
        acceptsMouseMovedEvents = true
        contentView = selectionView
        selectionView.frame = NSRect(origin: .zero, size: screenFrame.size)
    }

    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }

    override func mouseDown(with event: NSEvent) {
        startPoint = event.locationInWindow
    }

    override func mouseDragged(with event: NSEvent) {
        guard let start = startPoint else { return }
        selectionView.selectionRect = rect(from: start, to: event.locationInWindow)
        selectionView.needsDisplay = true
    }

    override func mouseUp(with event: NSEvent) {
        defer { onFinish = nil }
        guard let start = startPoint else { onFinish?(nil); return }
        let winRect = rect(from: start, to: event.locationInWindow)
        if winRect.width < 4 || winRect.height < 4 {
            onFinish?(nil)
            return
        }
        // Window frame == screen frame, so window coords → global by offset.
        let global = CGRect(x: frame.minX + winRect.minX,
                            y: frame.minY + winRect.minY,
                            width: winRect.width, height: winRect.height)
        onFinish?(global)
    }

    override func keyDown(with event: NSEvent) {
        if event.keyCode == 53 {   // Esc
            onFinish?(nil)
            onFinish = nil
        }
    }

    private func rect(from a: NSPoint, to b: NSPoint) -> NSRect {
        NSRect(x: min(a.x, b.x), y: min(a.y, b.y),
               width: abs(a.x - b.x), height: abs(a.y - b.y))
    }
}

final class SelectionView: NSView {
    var selectionRect: NSRect = .zero

    override func draw(_ dirtyRect: NSRect) {
        guard selectionRect.width > 0, selectionRect.height > 0 else { return }
        NSColor.systemBlue.withAlphaComponent(0.20).setFill()
        selectionRect.fill()
        NSColor.systemBlue.setStroke()
        let path = NSBezierPath(rect: selectionRect)
        path.lineWidth = 1.5
        path.stroke()
    }
}
