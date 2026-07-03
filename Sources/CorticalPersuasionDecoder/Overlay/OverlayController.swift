import AppKit
import SwiftUI

/// Owns the floating overlay panel that renders a `VerdictCard` next to the
/// user's selection. Borderless, non-activating (never steals focus), `.floating`
/// level. Auto-dismisses on any click, Esc, or after a timeout.
/// All methods must be called on the main thread (callers hop via `MainActor.run`).
final class OverlayController {
    private var panel: NSPanel?
    private var globalMonitor: Any?
    private var localMonitor: Any?
    private var dismissWork: DispatchWorkItem?

    /// `anchor` is a global AppKit point (bottom-left origin) near the selection.
    func show(verdict: Verdict, entry: TaxonomyEntry, anchor: CGPoint) {
        dismiss()

        let card = VerdictCard(title: entry.displayName,
                               brainRegion: entry.brainRegion,
                               mechanism: entry.mechanism,
                               confidence: verdict.confidence,
                               rationale: verdict.rationale)
        let hosting = NSHostingView(rootView: card)
        hosting.layoutSubtreeIfNeeded()
        let size = hosting.fittingSize

        let panel = NSPanel(contentRect: NSRect(origin: .zero, size: size),
                            styleMask: [.borderless, .nonactivatingPanel],
                            backing: .buffered, defer: false)
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = true
        panel.level = .floating
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .transient]
        panel.contentView = hosting
        panel.setFrameOrigin(position(for: size, near: anchor))
        panel.orderFrontRegardless()
        self.panel = panel

        installDismissMonitors()
        scheduleAutoDismiss(after: 9)
    }

    func dismiss() {
        dismissWork?.cancel(); dismissWork = nil
        if let globalMonitor { NSEvent.removeMonitor(globalMonitor); self.globalMonitor = nil }
        if let localMonitor { NSEvent.removeMonitor(localMonitor); self.localMonitor = nil }
        panel?.orderOut(nil)
        panel = nil
    }

    /// Place the card just below-and-right of the anchor, clamped to the screen.
    private func position(for size: CGSize, near anchor: CGPoint) -> CGPoint {
        let screen = NSScreen.screens.first { $0.frame.contains(anchor) } ?? NSScreen.main
        let visible = screen?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1440, height: 900)

        var x = anchor.x + 14
        var y = anchor.y - size.height - 14
        if x + size.width > visible.maxX { x = anchor.x - size.width - 14 }
        if x < visible.minX { x = visible.minX + 8 }
        if y < visible.minY { y = anchor.y + 14 }
        if y + size.height > visible.maxY { y = visible.maxY - size.height - 8 }
        return CGPoint(x: x, y: y)
    }

    private func installDismissMonitors() {
        globalMonitor = NSEvent.addGlobalMonitorForEvents(matching: [.leftMouseDown, .rightMouseDown]) { [weak self] _ in
            Task { @MainActor in self?.dismiss() }
        }
        localMonitor = NSEvent.addLocalMonitorForEvents(matching: [.leftMouseDown, .rightMouseDown, .keyDown]) { [weak self] event in
            if event.type == .keyDown && event.keyCode != 53 { return event }   // only Esc
            Task { @MainActor in self?.dismiss() }
            return event
        }
    }

    private func scheduleAutoDismiss(after seconds: TimeInterval) {
        let work = DispatchWorkItem { [weak self] in self?.dismiss() }
        dismissWork = work
        DispatchQueue.main.asyncAfter(deadline: .now() + seconds, execute: work)
    }
}
