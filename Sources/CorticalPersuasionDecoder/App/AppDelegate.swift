import AppKit
import os

let log = Logger(subsystem: "ai.zeonsystems.corticalpersuasiondecoder", category: "capture")

/// Writes to both stderr (visible when run from Terminal / the Xcode console) and
/// the unified log (visible in Console.app, filtered by the subsystem above).
func report(_ message: String) {
    FileHandle.standardError.write(Data(("[CPD] " + message + "\n").utf8))
    log.log("\(message, privacy: .public)")
}

final class AppDelegate: NSObject, NSApplicationDelegate {

    /// Primary Milestone 1 capture path: select text anywhere + ⌘C.
    private var pasteboardWatcher: PasteboardWatcher?

    /// Swappable classifier seam (Milestone 3). Mock for now; RemoteClassifier later.
    private let classifier: Classifier = MockClassifier()

    /// Menu-bar presence + region-capture trigger (Milestone 2).
    private var statusItem: NSStatusItem?
    private var regionSelector: RegionSelector?

    /// Floating verdict overlay (Milestone 4).
    private let overlay = OverlayController()

    // NOTE: `SelectionTextCapture` (Accessibility polling) and `HotkeyManager`
    // (Carbon global hotkey) still exist in the project but are intentionally
    // NOT started here. The AX path can't read web/article body text (it only
    // sees editable fields), and the ⌥⌘-style hotkeys collided with Chrome's
    // DevTools shortcut. The pasteboard watcher below is the reliable universal
    // path. Those files will be reused in later milestones.

    func applicationDidFinishLaunching(_ notification: Notification) {
        report("Cortical Persuasion Decoder — Milestone 1 (capture spike)")
        report("Interaction: select text in ANY app (Chrome article, PDF, native) and press ⌘C.")
        report("It reads what you copied — no special permissions, no hotkey conflicts.")
        report("Region OCR: click the 🧠 menu-bar icon → \"Capture Region\" to drag-select an area.")
        startPasteboardWatch()
        setupMenuBar()
    }

    // MARK: - Menu bar (Milestone 2 trigger)

    private func setupMenuBar() {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        item.button?.title = "🧠"
        let menu = NSMenu()
        let capture = NSMenuItem(title: "Capture Region (OCR → classify)",
                                 action: #selector(captureRegion), keyEquivalent: "")
        capture.target = self
        menu.addItem(capture)
        menu.addItem(.separator())
        let quit = NSMenuItem(title: "Quit Cortical Persuasion Decoder",
                              action: #selector(quitApp), keyEquivalent: "q")
        quit.target = self
        menu.addItem(quit)
        item.menu = menu
        statusItem = item
    }

    @objc private func quitApp() {
        NSApp.terminate(nil)
    }

    @objc private func captureRegion() {
        guard CGPreflightScreenCaptureAccess() else {
            report("⚠️  Screen Recording permission needed. Approve the dialog, then enable this app under")
            report("    System Settings ▸ Privacy & Security ▸ Screen Recording, and try again.")
            CGRequestScreenCaptureAccess()
            return
        }
        report("🖼️  Drag to select a region… (Esc to cancel)")
        let selector = RegionSelector()
        regionSelector = selector
        selector.begin { [weak self] rect in
            guard let self else { return }
            self.regionSelector = nil
            guard let rect else {
                report("🖼️  region capture cancelled.")
                return
            }
            Task { await self.captureRegionAndClassify(rect) }
        }
    }

    private func captureRegionAndClassify(_ rect: CGRect) async {
        do {
            let image = try await RegionCapture.capture(globalRect: rect)
            let url = URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("cpd_region.png")
            savePNG(image, to: url)
            report("🖼️  captured \(Int(rect.width))×\(Int(rect.height)) pt → \(url.path)")

            let text = try OCRService.recognizeText(in: image)
            guard !text.isEmpty else {
                report("🔤 OCR found no text in the selected region.")
                return
            }
            let oneLine = text.replacingOccurrences(of: "\n", with: " ⏎ ")
            report("🔤 OCR: \(oneLine)")
            // Anchor the card at the top-right corner of the captured region.
            await classifyAndPresent(text, anchor: CGPoint(x: rect.maxX, y: rect.maxY))
        } catch {
            report("⚠️  region/OCR failed: \(error.localizedDescription)")
        }
    }

    private func savePNG(_ image: CGImage, to url: URL) {
        let rep = NSBitmapImageRep(cgImage: image)
        if let data = rep.representation(using: .png, properties: [:]) {
            try? data.write(to: url)
        }
    }

    private func startPasteboardWatch() {
        let watcher = PasteboardWatcher { [weak self] text in
            self?.handleCapturedText(text)
        }
        watcher.start()
        pasteboardWatcher = watcher
        report("✂️  Ready — select any text and press ⌘C to capture it.")
    }

    /// Capture → classify → look up brain region → print the label + show overlay.
    private func handleCapturedText(_ text: String) {
        let oneLine = text.replacingOccurrences(of: "\n", with: " ⏎ ")
        let clipped = oneLine.count > 140 ? String(oneLine.prefix(140)) + "…" : oneLine
        report("✂️  copied (\(text.count) chars): \(clipped)")
        // The mouse is at the end of the just-made selection — a good anchor.
        let anchor = NSEvent.mouseLocation
        Task { await classifyAndPresent(text, anchor: anchor) }
    }

    private func classifyAndPresent(_ text: String, anchor: CGPoint) async {
        do {
            let verdict = try await classifier.classify(ClassificationInput(text: text))
            let entry = Taxonomy.entry(for: verdict.vector)
            let pct = Int((verdict.confidence * 100).rounded())
            report("🧠 \(entry.displayName)  →  \(entry.brainRegion)   [confidence \(pct)%]")
            report("   ↳ mechanism: \(entry.mechanism)")
            if let why = verdict.rationale {
                report("   ↳ why: \(why)")
            }
            await MainActor.run {
                overlay.show(verdict: verdict, entry: entry, anchor: anchor)
            }
        } catch {
            report("⚠️  classify failed: \(error.localizedDescription)")
        }
    }
}
