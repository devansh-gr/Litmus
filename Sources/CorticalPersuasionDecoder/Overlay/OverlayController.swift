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

    private var current: (title: String, mechanism: String, confidence: Double,
                          rationale: String?, mixture: [MixtureItem], uncertain: Bool)?

    /// `anchor` is a global AppKit point (bottom-left origin) near the selection.
    /// Returns a generation token to pass back to `updateRegions`.
    @discardableResult
    func show(verdict: Verdict, entry: TaxonomyEntry, anchor: CGPoint) -> Int {
        let mixture = verdict.alternatives
            .filter { $0.probability >= 0.08 }
            .prefix(2)
            .map { MixtureItem(label: Taxonomy.entry(for: $0.vector).displayName,
                               pct: Int(($0.probability * 100).rounded())) }
        current = (entry.displayName, entry.mechanism, verdict.confidence,
                   verdict.rationale, Array(mixture), verdict.uncertain)
        return showCard(awaiting: false, anchor: anchor)
    }

    /// Re-present the LAST verdict — used by a menu-triggered deep-scan, since
    /// opening the menu bar dismisses the on-screen card. Returns a fresh token to
    /// drive `beginBrainScan`/`updateProfile`, or nil if there's nothing to show.
    @discardableResult
    func reshowLastVerdict(anchor: CGPoint) -> Int? {
        guard current != nil else { return nil }
        return showCard(awaiting: true, anchor: anchor)
    }

    /// Present a verdict panel from the stored `current`. `awaiting` shows the
    /// brain-scan spinner state immediately.
    private func showCard(awaiting: Bool, anchor: CGPoint) -> Int {
        dismiss()
        generation += 1

        let hosting = NSHostingView(rootView: makeCard(profile: nil, failed: false, awaiting: awaiting))
        hosting.layoutSubtreeIfNeeded()
        let size = hosting.fittingSize

        let panel = NSPanel(contentRect: NSRect(origin: .zero, size: size),
                            styleMask: [.borderless, .nonactivatingPanel],
                            backing: .buffered, defer: false)
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = true
        panel.level = .floating
        // NOT .transient: a transient panel is hidden whenever its owning app is
        // inactive, and this LSUIElement agent is essentially never active — that
        // silently hid the verdict card ("scans, but nothing pops up").
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        panel.contentView = hosting
        panel.setFrameOrigin(position(for: size, near: anchor))
        panel.orderFrontRegardless()

        self.panel = panel
        self.hosting = hosting

        let f = panel.frame
        report("🃏 card presented: visible=\(panel.isVisible) appActive=\(NSApp.isActive) "
             + "at \(Int(f.minX)),\(Int(f.minY)) \(Int(f.width))×\(Int(f.height))")

        installDismissMonitors()
        // While awaiting a deep scan the card must outlive the brain map, which can
        // take ~2 min on a cold TRIBE reload (server timeout is 300s). A fixed 60s
        // here dismissed the card first, so updateProfile's `panel != nil` guard
        // then silently dropped the result — "deep scan does nothing".
        scheduleAutoDismiss(after: awaiting ? Self.awaitingTimeout : Self.normalTimeout)
        return generation
    }

    /// Auto-dismiss windows: short for a plain verdict, long enough to outlast a
    /// cold brain-map (>300s server timeout) while a deep scan is in flight.
    private static let normalTimeout: TimeInterval = 60
    private static let awaitingTimeout: TimeInterval = 360

    /// Switch the current card into the "scanning cortex" state (spinner). Called
    /// when the user opts into a deep scan from the menu.
    func beginBrainScan(token: Int) {
        guard token == generation, let hosting else { return }
        hosting.rootView = makeCard(profile: nil, failed: false, awaiting: true)
        resizeToFit()
        // A deep scan can be triggered on an already-open card (whose short timer is
        // running) — extend it so the map isn't dropped mid-flight.
        scheduleAutoDismiss(after: Self.awaitingTimeout)
    }

    /// Fill in the cortical impact profile once TRIBE v2 returns. `token` must
    /// match the generation from `show`, else the card has moved on and we drop it.
    func updateProfile(_ profile: [CorticalSystem]?, failed: Bool, token: Int) {
        guard token == generation, let hosting, panel != nil else { return }
        hosting.rootView = makeCard(profile: profile, failed: failed, awaiting: false)
        resizeToFit()
        // Result is in — revert the long deep-scan window to the normal linger.
        scheduleAutoDismiss(after: Self.normalTimeout)
    }

    /// Re-fit the panel to the current card, keeping the top-left corner pinned.
    private func resizeToFit() {
        guard let hosting, let panel else { return }
        hosting.layoutSubtreeIfNeeded()
        let newSize = hosting.fittingSize
        let frame = panel.frame
        let topY = frame.maxY
        var newFrame = NSRect(x: frame.minX, y: topY - newSize.height,
                              width: newSize.width, height: newSize.height)
        // A deep scan makes the card taller; without re-clamping it can spill below
        // the screen and clip the brain-profile rows. Keep it on-screen.
        let screen = NSScreen.screens.first { $0.frame.intersects(newFrame) } ?? NSScreen.main
        if let visible = screen?.visibleFrame {
            if newFrame.maxX > visible.maxX { newFrame.origin.x = visible.maxX - newFrame.width - 8 }
            if newFrame.minX < visible.minX { newFrame.origin.x = visible.minX + 8 }
            if newFrame.minY < visible.minY { newFrame.origin.y = visible.minY + 8 }
            if newFrame.maxY > visible.maxY { newFrame.origin.y = visible.maxY - newFrame.height - 8 }
        }
        panel.setFrame(newFrame, display: true)
    }

    private func makeCard(profile: [CorticalSystem]?, failed: Bool, awaiting: Bool) -> VerdictCard {
        let c = current ?? ("", "", 0, nil, [], false)
        return VerdictCard(
            title: c.title,
            mechanism: c.mechanism,
            confidence: c.confidence,
            rationale: c.rationale,
            mixture: c.mixture,
            uncertain: c.uncertain,
            profile: profile,
            regionsFailed: failed,
            awaitingBrain: awaiting
        )
    }

    /// Show a transient notice (e.g. server offline) in the same floating style.
    func showNotice(_ message: String, anchor: CGPoint) {
        dismiss()
        generation += 1
        current = nil

        let hosting = NSHostingView(rootView: NoticeCard(message: message))
        hosting.layoutSubtreeIfNeeded()
        let size = hosting.fittingSize

        let panel = NSPanel(contentRect: NSRect(origin: .zero, size: size),
                            styleMask: [.borderless, .nonactivatingPanel],
                            backing: .buffered, defer: false)
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = true
        panel.level = .floating
        // NOT .transient: a transient panel is hidden whenever its owning app is
        // inactive, and this LSUIElement agent is essentially never active — that
        // silently hid the verdict card ("scans, but nothing pops up").
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        panel.contentView = hosting
        panel.setFrameOrigin(position(for: size, near: anchor))
        panel.orderFrontRegardless()

        self.panel = panel
        self.hosting = nil   // notices don't get profile updates
        installDismissMonitors()
        scheduleAutoDismiss(after: 6)
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
        // Cancel any pending dismissal first: re-arming (deep scan extends, then the
        // result reverts) would otherwise leave the earlier, shorter timer live and
        // it would dismiss the card out from under the new deadline.
        dismissWork?.cancel()
        let work = DispatchWorkItem { [weak self] in self?.dismiss() }
        dismissWork = work
        DispatchQueue.main.asyncAfter(deadline: .now() + seconds, execute: work)
    }
}
