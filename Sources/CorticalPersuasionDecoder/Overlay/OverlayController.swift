import AppKit
import SwiftUI

/// Owns the floating overlay panel that renders a `VerdictCard` next to the
/// user's selection. Borderless, non-activating (never steals focus), `.floating`
/// level. Auto-dismisses on any click, Esc, or after a timeout.
///
/// The card is shown IMMEDIATELY with the LLM verdict (~3s) and the cortical map
/// filled in later via `updateRegions` (~30s+), because TRIBE v2 is far too slow
/// to sit in the interactive path.
///
/// All methods must be called on the main thread (callers hop via `MainActor.run`).
final class OverlayController {
    private var panel: NSPanel?
    private var hosting: NSHostingView<VerdictCard>?
    private var globalMonitor: Any?
    private var localMonitor: Any?
    private var dismissWork: DispatchWorkItem?

    /// Token identifying the currently shown verdict, so a late-arriving brain map
    /// from a previous selection cannot overwrite a newer card.
    private var generation = 0

    private var current: (title: String, mechanism: String, confidence: Double, rationale: String?)?

    /// `anchor` is a global AppKit point (bottom-left origin) near the selection.
    /// Returns a generation token to pass back to `updateRegions`.
    @discardableResult
    func show(verdict: Verdict, entry: TaxonomyEntry, anchor: CGPoint) -> Int {
        dismiss()
        generation += 1

        current = (entry.displayName, entry.mechanism, verdict.confidence, verdict.rationale)
        let card = makeCard(regions: nil, failed: false, awaiting: false)

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
        self.hosting = hosting

        installDismissMonitors()
        // Generous timeout: the brain map can take ~30s+ to arrive.
        scheduleAutoDismiss(after: 60)
        return generation
    }

    /// Switch the current card into the "scanning cortex" state (spinner). Called
    /// when the user opts into a deep scan from the menu.
    func beginBrainScan(token: Int) {
        guard token == generation, let hosting else { return }
        hosting.rootView = makeCard(regions: nil, failed: false, awaiting: true)
        resizeToFit()
    }

    /// Fill in the cortical map once TRIBE v2 returns. `token` must match the
    /// generation from `show`, otherwise the card has moved on and we drop it.
    func updateRegions(_ regions: [BrainRegion]?, failed: Bool, token: Int) {
        guard token == generation, let hosting, let panel else { return }
        _ = panel
        hosting.rootView = makeCard(regions: regions, failed: failed, awaiting: false)
        resizeToFit()
    }

    /// Re-fit the panel to the current card, keeping the top-left corner pinned.
    private func resizeToFit() {
        guard let hosting, let panel else { return }
        hosting.layoutSubtreeIfNeeded()
        let newSize = hosting.fittingSize
        let frame = panel.frame
        let topY = frame.maxY
        panel.setFrame(
            NSRect(x: frame.minX, y: topY - newSize.height,
                   width: newSize.width, height: newSize.height),
            display: true
        )
    }

    private func makeCard(regions: [BrainRegion]?, failed: Bool, awaiting: Bool) -> VerdictCard {
        let c = current ?? ("", "", 0, nil)
        return VerdictCard(
            title: c.title,
            mechanism: c.mechanism,
            confidence: c.confidence,
            rationale: c.rationale,
            regions: regions,
            regionsFailed: failed,
            awaitingBrain: awaiting
        )
    }

    func dismiss() {
        dismissWork?.cancel(); dismissWork = nil
        if let globalMonitor { NSEvent.removeMonitor(globalMonitor); self.globalMonitor = nil }
        if let localMonitor { NSEvent.removeMonitor(localMonitor); self.localMonitor = nil }
        panel?.orderOut(nil)
        panel = nil
        hosting = nil
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
