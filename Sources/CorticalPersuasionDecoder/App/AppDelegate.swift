import AppKit
import os

let log = Logger(subsystem: "ai.zeonsystems.corticalpersuasiondecoder", category: "capture")

/// Writes to stderr (Terminal/Xcode console), the unified log (Console.app), and a
/// plain log file — the file is the only one visible when launched via `open`.
let cpdLogFile = (NSHomeDirectory() as NSString)
    .appendingPathComponent("Library/Logs/CorticalPersuasionDecoder.log")

/// Serialises the file append so concurrent report() calls from background Tasks
/// and the main thread can't interleave / garble log lines.
private let cpdLogQueue = DispatchQueue(label: "ai.zeonsystems.cpd.log")

func report(_ message: String) {
    FileHandle.standardError.write(Data(("[CPD] " + message + "\n").utf8))
    log.log("\(message, privacy: .public)")
    let line = "[\(Date())] \(message)\n"
    cpdLogQueue.async {
        let url = URL(fileURLWithPath: cpdLogFile)
        if let fh = try? FileHandle(forWritingTo: url) {
            fh.seekToEndOfFile(); try? fh.write(contentsOf: Data(line.utf8)); try? fh.close()
        } else {
            try? Data(line.utf8).write(to: url)
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate, NSMenuDelegate {

    /// A completed scan, kept so the menu can re-show it, copy it, or list history.
    private struct Scan { let text: String; let entry: TaxonomyEntry; let verdict: Verdict }
    private var lastScan: Scan?
    private var history: [Scan] = []          // most-recent-first, capped at 8
    private weak var lastItem: NSMenuItem?
    private weak var copyItem: NSMenuItem?
    private weak var recentItem: NSMenuItem?

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
        menu.autoenablesItems = false   // we manage enabled state in menuNeedsUpdate
        menu.delegate = self

        let last = NSMenuItem(title: "No scans yet", action: #selector(showLastScan), keyEquivalent: "")
        last.target = self
        menu.addItem(last)
        lastItem = last

        let copy = NSMenuItem(title: "Copy last verdict", action: #selector(copyLastVerdict), keyEquivalent: "c")
        copy.target = self
        menu.addItem(copy)
        copyItem = copy

        let clip = NSMenuItem(title: "Analyze clipboard", action: #selector(analyzeClipboard), keyEquivalent: "")
        clip.target = self
        menu.addItem(clip)

        let recent = NSMenuItem(title: "Recent", action: nil, keyEquivalent: "")
        recent.submenu = NSMenu()
        menu.addItem(recent)
        recentItem = recent

        menu.addItem(.separator())
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

    // MARK: - Dynamic menu (last verdict + recent history)

    /// Refresh the last-verdict / recent items each time the menu opens.
    func menuNeedsUpdate(_ menu: NSMenu) {
        if let s = lastScan {
            lastItem?.title = "Show: \(s.entry.displayName) · \(pctString(s.verdict))"
            lastItem?.isEnabled = true
            copyItem?.isEnabled = true
        } else {
            lastItem?.title = "No scans yet"
            lastItem?.isEnabled = false
            copyItem?.isEnabled = false
        }
        let sub = NSMenu()
        sub.autoenablesItems = false
        if history.isEmpty {
            let empty = NSMenuItem(title: "— none —", action: nil, keyEquivalent: "")
            empty.isEnabled = false
            sub.addItem(empty)
        } else {
            for (i, s) in history.enumerated() {
                let it = NSMenuItem(title: "\(s.entry.displayName) · \(pctString(s.verdict))  —  \(snippet(s.text))",
                                    action: #selector(showHistoryItem(_:)), keyEquivalent: "")
                it.target = self
                it.tag = i
                sub.addItem(it)
            }
        }
        recentItem?.submenu = sub
        recentItem?.isEnabled = !history.isEmpty
    }

    private func pctString(_ v: Verdict) -> String { "\(Int((v.confidence * 100).rounded()))%" }

    private func snippet(_ s: String) -> String {
        let one = s.replacingOccurrences(of: "\n", with: " ").trimmingCharacters(in: .whitespaces)
        return one.count > 32 ? String(one.prefix(32)) + "…" : one
    }

    /// Record a completed scan for the menu (last + history). Main thread.
    private func recordScan(text: String, entry: TaxonomyEntry, verdict: Verdict) {
        let scan = Scan(text: text, entry: entry, verdict: verdict)
        lastScan = scan
        history.insert(scan, at: 0)
        if history.count > 8 { history.removeLast() }
        statusItem?.button?.toolTip = "Last: \(entry.displayName) · \(pctString(verdict))"
    }

    @objc private func showLastScan() {
        guard let s = lastScan else { return }
        overlay.show(verdict: s.verdict, entry: s.entry, anchor: NSEvent.mouseLocation)
        lastCapturedText = s.text
    }

    @objc private func showHistoryItem(_ sender: NSMenuItem) {
        guard history.indices.contains(sender.tag) else { return }
        let s = history[sender.tag]
        overlay.show(verdict: s.verdict, entry: s.entry, anchor: NSEvent.mouseLocation)
        lastCapturedText = s.text
    }

    @objc private func copyLastVerdict() {
        guard let s = lastScan else { return }
        var text = "\(s.entry.displayName) (\(pctString(s.verdict))) — \(s.entry.mechanism)"
        if let r = s.verdict.rationale { text += "\nWhy: \(r)" }
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(text, forType: .string)
        report("📋 copied last verdict to clipboard.")
    }

    @objc private func analyzeClipboard() {
        guard let text = NSPasteboard.general.string(forType: .string), !text.isEmpty else {
            overlay.showNotice("Clipboard is empty — copy some text first.", anchor: NSEvent.mouseLocation)
            return
        }
        analyze(text, anchor: NSEvent.mouseLocation, source: "clipboard")
    }

    @objc private func quitApp() {
        NSApp.terminate(nil)
    }

    @objc private func togglePause() {
        captureEnabled.toggle()
        pauseItem?.title = captureEnabled ? "Pause capture" : "Resume capture"
        refreshStatusIcon()
        report(captureEnabled ? "▶️  capture resumed." : "⏸  capture paused.")
    }

    @objc private func captureRegion() {
        // Anchor notices near the cursor (the menu-bar click point).
        let anchor = NSEvent.mouseLocation
        guard Permissions.isScreenRecordingTrusted() else {
            report("⚠️  Screen Recording permission needed for Capture Region.")
            report("    Enable this app under System Settings ▸ Privacy & Security ▸ Screen Recording, then REOPEN it.")
            // Critical UX: without an on-screen notice this looked like a dead click.
            overlay.showNotice("Screen Recording is off. Enable it for this app in System Settings, then reopen the app.",
                               anchor: anchor)
            Permissions.requestScreenRecording()       // system dialog (first request only)
            Permissions.openScreenRecordingSettings()   // jump straight to the pane
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
        let anchor = CGPoint(x: rect.maxX, y: rect.maxY)
        // Let the selection overlay finish disappearing before the screenshot, so we
        // capture the underlying text rather than our blue dimming layer.
        try? await Task.sleep(nanoseconds: 120_000_000)
        do {
            let image = try await RegionCapture.capture(globalRect: rect)
            let url = URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("cpd_region.png")
            savePNG(image, to: url)
            report("🖼️  captured \(Int(rect.width))×\(Int(rect.height)) pt → \(url.path)")

            let text = try OCRService.recognizeText(in: image)
            guard !text.isEmpty else {
                report("🔤 OCR found no text in the selected region.")
                await MainActor.run { self.overlay.showNotice("No readable text in that region.", anchor: anchor) }
                return
            }
            // Route through the same guards as ⌘B (pause / secrets / trivial / busy),
            // anchored at the region's top-right corner.
            await MainActor.run {
                self.analyze(text, anchor: anchor, source: "OCR")
            }
        } catch {
            report("⚠️  region/OCR failed: \(error.localizedDescription)")
            await MainActor.run {
                self.overlay.showNotice("Capture failed: \(error.localizedDescription)", anchor: anchor)
            }
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
        guard !isBusy else { report("⏳ busy — ⌘B ignored (a scan is running)."); return }
        // Instant feedback the moment ⌘B is pressed — BEFORE the ~0.3s copy + classify —
        // so the user always sees something is happening. Anchor is captured now (the
        // mouse may move during the scan).
        let anchor = NSEvent.mouseLocation
        overlay.showNotice("Analyzing…", anchor: anchor)
        // Small delay so the physical ⌘B key-up doesn't collide with the synthetic ⌘C.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) { [weak self] in
            guard let self else { return }
            guard let text = PasteboardCapture.selectedTextViaCopy(), !text.isEmpty else {
                report("⌘B: no text selected.")
                self.overlay.showNotice("No text selected — highlight something first.", anchor: anchor)
                return
            }
            self.analyze(text, anchor: anchor, source: "copied")
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
        overlay.showNotice("Analyzing…", anchor: anchor)   // instant feedback so a 1–4s scan doesn't look dead
        isBusy = true
        refreshStatusIcon()
        Task {
            await classifyAndPresent(trimmed, anchor: anchor)
            await MainActor.run { self.isBusy = false; self.refreshStatusIcon() }
        }
    }

    /// Reflect state in the menu-bar icon: idle 🧠, paused 🧠⏸, scanning 🧠…
    private func refreshStatusIcon() {
        statusItem?.button?.title = isBusy ? "🧠…" : (captureEnabled ? "🧠" : "🧠⏸")
    }

    private func classifyAndPresent(_ text: String, anchor: CGPoint) async {
        let verdict: Verdict
        do {
            verdict = try await classifier.classify(ClassificationInput(text: text))
        } catch RemoteClassifierError.neutral {
            report("🧠 neutral — no persuasion vector detected.")
            await MainActor.run { overlay.showNotice("No manipulation detected ✓", anchor: anchor) }
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
            // Everything else (bad HTTP status, JSON decode, server error) used to
            // only log — the "Analyzing…" notice just faded and ⌘B looked dead.
            report("⚠️  classify failed: \(error.localizedDescription)")
            await MainActor.run {
                overlay.showNotice("Couldn't analyze that — \(error.localizedDescription)", anchor: anchor)
            }
            return
        }

        // No confidence threshold: always show the verdict (even weak/low-confidence),
        // so ⌘B never silently no-ops. The confidence number itself conveys strength.
        let entry = Taxonomy.entry(for: verdict.vector)
        let pct = Int((verdict.confidence * 100).rounded())
        report("🧠 \(entry.displayName)   [confidence \(pct)%]")
        report("   ↳ mechanism: \(entry.mechanism)")

        // Show the card immediately with the LLM verdict. The cortical brain map
        // is OPT-IN (it takes ~2 min), triggered from the 🧠 menu — we do NOT
        // auto-run it on every selection.
        // Show the card AND record the deep-scan state on the main actor together —
        // deepScanCortex reads these on the main thread, so writing them off-thread
        // (as before) was a data race.
        await MainActor.run {
            let token = overlay.show(verdict: verdict, entry: entry, anchor: anchor)
            self.lastCapturedText = text
            self.lastOverlayToken = token
            self.recordScan(text: text, entry: entry, verdict: verdict)
        }
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
        // The verdict card was dismissed when the menu opened, so re-present it to
        // host the result. If there's nothing to re-present (a notice cleared it),
        // abort with a clear message instead of using a stale token whose result
        // would be silently dropped by the generation guard.
        guard let token = overlay.reshowLastVerdict(anchor: NSEvent.mouseLocation) else {
            report("🧠 deep scan: nothing to scan — press ⌘B on some text first.")
            overlay.showNotice("Nothing to deep-scan yet — press ⌘B on text first.",
                               anchor: NSEvent.mouseLocation)
            return
        }
        report("🧠 deep scan: modelling cortex for last selection (~30s)…")
        overlay.beginBrainScan(token: token)
        isBusy = true
        refreshStatusIcon()
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
            await MainActor.run { self.isBusy = false; self.refreshStatusIcon() }
        }
    }
}
