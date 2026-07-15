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

    /// Detection: LLM (remote) by default, or the offline mock. See Config.
    private let classifier: Classifier = Config.makeClassifier()

    /// Interpretation: TRIBE v2 cortical map. Nil when running on the mock.
    private let brainMapper: RemoteClassifier? = Config.makeBrainMapper()

    /// The last analysed selection + its overlay generation, for opt-in deep scan.
    private var lastCapturedText: String?
    private var lastOverlayToken: Int = 0

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
        let deepScan = NSMenuItem(title: "Deep-scan cortex (last selection)",
                                  action: #selector(deepScanCortex), keyEquivalent: "")
        deepScan.target = self
        menu.addItem(deepScan)
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
        let verdict: Verdict
        do {
            verdict = try await classifier.classify(ClassificationInput(text: text))
        } catch RemoteClassifierError.neutral {
            report("🧠 neutral — no persuasion vector detected.")
            return
        } catch {
            report("⚠️  classify failed: \(error.localizedDescription)")
            return
        }

        guard verdict.confidence >= Config.confidenceThreshold else {
            let pct = Int((verdict.confidence * 100).rounded())
            report("🧠 \(verdict.vector.rawValue) at \(pct)% — below threshold, suppressed.")
            return
        }

        let entry = Taxonomy.entry(for: verdict.vector)
        let pct = Int((verdict.confidence * 100).rounded())
        report("🧠 \(entry.displayName)   [confidence \(pct)%]")
        report("   ↳ mechanism: \(entry.mechanism)")

        // Show the card immediately with the LLM verdict. The cortical brain map
        // is OPT-IN (it takes ~2 min), triggered from the 🧠 menu — we do NOT
        // auto-run it on every selection.
        let token = await MainActor.run {
            overlay.show(verdict: verdict, entry: entry, anchor: anchor)
        }
        lastCapturedText = text
        lastOverlayToken = token
    }

    // MARK: - Opt-in deep cortical scan (Milestone 5, slow path)

    @objc private func deepScanCortex() {
        guard let brainMapper, let text = lastCapturedText else {
            report("🧠 deep scan: nothing captured yet — select text + ⌘C first.")
            return
        }
        let token = lastOverlayToken
        report("🧠 deep scan: modelling cortex for last selection (~2 min)…")
        overlay.beginBrainScan(token: token)
        Task {
            do {
                let regions = try await brainMapper.brainMap(for: text)
                let top = regions.prefix(3)
                    .map { "\($0.region) (z\(String(format: "%+.1f", $0.activationZ)))" }
                    .joined(separator: ", ")
                report("   ↳ cortex: \(top)")
                await MainActor.run { overlay.updateRegions(regions, failed: false, token: token) }
            } catch {
                report("   ↳ cortical map unavailable: \(error.localizedDescription)")
                await MainActor.run { overlay.updateRegions(nil, failed: true, token: token) }
            }
        }
    }
}
