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
        NSApp.activate(ignoringOtherApps: true)
        window.makeKeyAndOrderFront(nil)
        window.orderFrontRegardless()
    }
}

final class SelectionWindow: NSWindow {
    var onFinish: ((CGRect?) -> Void)? {
        get { view.onFinish }
        set { view.onFinish = newValue }
    }
    private let view = SelectionView()

    init(screenFrame: NSRect) {
        super.init(contentRect: screenFrame, styleMask: .borderless, backing: .buffered, defer: false)
        isOpaque = false
        backgroundColor = NSColor.black.withAlphaComponent(0.25)
        level = .screenSaver
        collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        ignoresMouseEvents = false
        contentView = view
        view.frame = NSRect(origin: .zero, size: screenFrame.size)
    }

    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }

    override func keyDown(with event: NSEvent) {
        if event.keyCode == 53 { view.cancel() }   // Esc
    }
}

final class SelectionView: NSView {
    var onFinish: ((CGRect?) -> Void)?
    private var startPoint: NSPoint?
    private var currentRect: NSRect = .zero

    override var acceptsFirstResponder: Bool { true }
    // Critical: without this, an .accessory app's overlay eats the first click as
    // window activation and the drag is lost.
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    override func resetCursorRects() {
        addCursorRect(bounds, cursor: .crosshair)
    }

    override func mouseDown(with event: NSEvent) {
        startPoint = convert(event.locationInWindow, from: nil)
        currentRect = .zero
        needsDisplay = true
    }

    override func mouseDragged(with event: NSEvent) {
        guard let start = startPoint else { return }
        currentRect = rect(from: start, to: convert(event.locationInWindow, from: nil))
        needsDisplay = true
    }

    override func mouseUp(with event: NSEvent) {
        defer { reset() }
        guard let start = startPoint else { onFinish?(nil); return }
        let local = rect(from: start, to: convert(event.locationInWindow, from: nil))
        if local.width < 4 || local.height < 4 {
            onFinish?(nil)
            return
        }
        // View fills the window (bottom-left origin); global = window origin + local.
        let origin = window?.frame.origin ?? .zero
        onFinish?(CGRect(x: origin.x + local.minX, y: origin.y + local.minY,
                         width: local.width, height: local.height))
    }

    func cancel() {
        onFinish?(nil)
        reset()
    }

    private func reset() {
        onFinish = nil
        startPoint = nil
        currentRect = .zero
    }

    private func rect(from a: NSPoint, to b: NSPoint) -> NSRect {
        NSRect(x: min(a.x, b.x), y: min(a.y, b.y),
               width: abs(a.x - b.x), height: abs(a.y - b.y))
    }

    override func draw(_ dirtyRect: NSRect) {
        let hint = "Drag to select a region  ·  Esc to cancel"
        hint.draw(at: NSPoint(x: 24, y: bounds.height - 48), withAttributes: [
            .font: NSFont.systemFont(ofSize: 16, weight: .semibold),
            .foregroundColor: NSColor.white,
        ])

        guard currentRect.width > 0, currentRect.height > 0 else { return }
        NSColor.systemBlue.withAlphaComponent(0.18).setFill()
        currentRect.fill()
        NSColor.systemBlue.setStroke()
        let path = NSBezierPath(rect: currentRect)
        path.lineWidth = 2
        path.stroke()
    }
}
