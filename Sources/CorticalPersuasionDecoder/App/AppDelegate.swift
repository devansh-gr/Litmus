import AppKit
import os

let log = Logger(subsystem: "ai.zeonsystems.corticalpersuasiondecoder", category: "capture")

/// Writes to stderr (Terminal/Xcode console), the unified log (Console.app), and a
/// plain log file — the file is the only one visible when launched via `open`.
let cpdLogFile = (NSHomeDirectory() as NSString)
    .appendingPathComponent("Library/Logs/CorticalPersuasionDecoder.log")

func report(_ message: String) {
    FileHandle.standardError.write(Data(("[CPD] " + message + "\n").utf8))
    log.log("\(message, privacy: .public)")
    let line = "[\(Date())] \(message)\n"
    let url = URL(fileURLWithPath: cpdLogFile)
    if let fh = try? FileHandle(forWritingTo: url) {
        fh.seekToEndOfFile(); try? fh.write(contentsOf: Data(line.utf8)); try? fh.close()
    } else {
        try? Data(line.utf8).write(to: url)
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {

    /// Primary capture path: select text anywhere, press ⌘B to analyze it.
    private var hotkeyMonitor: HotkeyMonitor?
    private var accessibilityPollTimer: Timer?

    /// Detection: LLM (remote) by default, or the offline mock. See Config.
    private let classifier: Classifier = Config.makeClassifier()

    /// Interpretation: TRIBE v2 cortical map. Nil when running on the mock.
    private let brainMapper: RemoteClassifier? = Config.makeBrainMapper()

    /// The last analysed selection + its overlay generation, for opt-in deep scan.
    private var lastCapturedText: String?
    private var lastOverlayToken: Int = 0

    /// Capture on/off (menu-toggled) so the user can stop analysing every copy.
    private var captureEnabled = true
    private weak var pauseItem: NSMenuItem?

    /// True while a classify or deep-scan is running — serialises GPU work (MLX
    /// classify vs PyTorch brain-map both use Metal) and prevents request backlog.
    private var isBusy = false

    /// Menu-bar presence + region-capture trigger (Milestone 2).
    private var statusItem: NSStatusItem?
    private var regionSelector: RegionSelector?

    /// Floating verdict overlay (Milestone 4).
    private let overlay = OverlayController()

    // Capture is triggered by ⌘B: an NSEvent global hotkey (non-consuming, so ⌘B
    // still bolds in editors) synthesizes a copy to grab the selection — which
    // reads web/article body the Accessibility API can't. Needs a one-time
    // Accessibility grant (for the global monitor + synthetic keystroke).

    func applicationDidFinishLaunching(_ notification: Notification) {
        report("Cortical Persuasion Decoder")
        report("Interaction: select text in ANY app (Chrome/Safari article, PDF, X…) and press ⌘B.")
        report("Region OCR: 🧠 menu-bar icon → \"Capture Region\" to drag-select an area.")
        setupMenuBar()
        ensureAccessibilityThenStart()
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
        let pause = NSMenuItem(title: "Pause capture", action: #selector(togglePause), keyEquivalent: "")
        pause.target = self
        menu.addItem(pause)
        pauseItem = pause
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

    @objc private func togglePause() {
        captureEnabled.toggle()
        pauseItem?.title = captureEnabled ? "Pause capture" : "Resume capture"
        statusItem?.button?.title = captureEnabled ? "🧠" : "🧠⏸"
        report(captureEnabled ? "▶️  capture resumed." : "⏸  capture paused.")
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
            // Route through the same guards as ⌘B (pause / secrets / trivial / busy),
            // anchored at the region's top-right corner.
            await MainActor.run {
                self.analyze(text, anchor: CGPoint(x: rect.maxX, y: rect.maxY), source: "OCR")
            }
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

    // MARK: - ⌘B hotkey (needs Accessibility)

    private func ensureAccessibilityThenStart() {
        if Permissions.isAccessibilityTrusted() {
            startHotkey()
            return
        }
        report("⚠️  Accessibility permission needed for the ⌘B hotkey.")
        report("    Approve the dialog, then enable this app under")
        report("    System Settings ▸ Privacy & Security ▸ Accessibility. Waiting…")
        Permissions.promptForAccessibility()
        Permissions.openAccessibilitySettings()
        accessibilityPollTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] timer in
            guard Permissions.isAccessibilityTrusted() else { return }
            timer.invalidate()
            self?.accessibilityPollTimer = nil
            report("✅ Accessibility granted.")
            self?.startHotkey()
        }
    }

    private func startHotkey() {
        let monitor = HotkeyMonitor { [weak self] in self?.handleHotkey() }
        monitor.start()
        hotkeyMonitor = monitor
        report("✂️  Ready — select any text and press ⌘B to analyze it.")
        // Ad-hoc builds: the Accessibility grant is tied to the code hash, so a
        // rebuild can silently break ⌘B. If that happens, re-toggle this app under
        // System Settings ▸ Privacy & Security ▸ Accessibility.
        report("   (if ⌘B ever stops working after a rebuild, re-grant Accessibility.)")
    }

    private func handleHotkey() {
        guard captureEnabled else { return }
        // Small delay so the physical ⌘B key-up doesn't collide with the synthetic ⌘C.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) { [weak self] in
            guard let self else { return }
            guard let text = PasteboardCapture.selectedTextViaCopy(), !text.isEmpty else {
                report("⌘B: no text selected.")
                return
            }
            self.analyze(text, anchor: NSEvent.mouseLocation, source: "copied")
        }
    }

    /// Shared gate for BOTH ⌘B and region-OCR: honor pause, skip secrets and
    /// trivial/blank text, avoid overlapping GPU work, then classify. Must run on
    /// the main thread. (Fixes region-bypass, blank input, Metal overlap, backlog.)
    private func analyze(_ text: String, anchor: CGPoint, source: String) {
        guard captureEnabled else { report("⏸  paused — \(source) ignored."); return }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count >= 3 else {
            report("· \(source): too short to analyze — skipped."); return
        }
        guard !SensitiveText.looksSensitive(text) else {
            report("🔒 \(source) looks like a secret (password/token/card) — skipped."); return
        }
        guard !isBusy else {
            report("⏳ busy — \(source) ignored (a scan is running)."); return
        }
        let oneLine = trimmed.replacingOccurrences(of: "\n", with: " ⏎ ")
        let clipped = oneLine.count > 140 ? String(oneLine.prefix(140)) + "…" : oneLine
        report("✂️  \(source) (\(trimmed.count) chars): \(clipped)")
        isBusy = true
        Task {
            await classifyAndPresent(trimmed, anchor: anchor)
            await MainActor.run { self.isBusy = false }
        }
    }

    private func classifyAndPresent(_ text: String, anchor: CGPoint) async {
        let verdict: Verdict
        do {
            verdict = try await classifier.classify(ClassificationInput(text: text))
        } catch RemoteClassifierError.neutral {
            report("🧠 neutral — no persuasion vector detected.")
            return
        } catch let urlError as URLError {
            // A timeout means the server is up but the model is still warming —
            // NOT offline. Only connection failures are truly "offline".
            let message: String
            switch urlError.code {
            case .timedOut:
                message = "Model still warming up — press ⌘B again in a moment."
            case .cannotConnectToHost, .cannotFindHost,
                 .networkConnectionLost, .cannotLoadFromNetwork:
                message = "Inference server offline.\nStart it:  server/server.py"
            default:
                message = "Inference server error (\(urlError.code.rawValue))."
            }
            report("⚠️  \(message)")
            await MainActor.run { overlay.showNotice(message, anchor: anchor) }
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
        guard !isBusy else {
            report("⏳ busy — wait for the current scan to finish."); return
        }
        guard let brainMapper, let text = lastCapturedText else {
            report("🧠 deep scan: nothing captured yet — select text + ⌘B first.")
            return
        }
        let token = lastOverlayToken
        report("🧠 deep scan: modelling cortex for last selection (~2 min)…")
        overlay.beginBrainScan(token: token)
        isBusy = true
        Task {
            do {
                let profile = try await brainMapper.brainMap(for: text)
                let summary = profile
                    .map { "\($0.system.split(separator: " (").first.map(String.init) ?? $0.system): \($0.level)" }
                    .joined(separator: " · ")
                report("   ↳ cortical impact: \(summary)")
                await MainActor.run { overlay.updateProfile(profile, failed: false, token: token) }
            } catch {
                report("   ↳ cortical map unavailable: \(error.localizedDescription)")
                await MainActor.run { overlay.updateProfile(nil, failed: true, token: token) }
            }
            await MainActor.run { self.isBusy = false }
        }
    }
}
