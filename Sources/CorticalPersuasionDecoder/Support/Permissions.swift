import AppKit
import ApplicationServices
import CoreGraphics

/// Accessibility permission — needed by the ⌘B hotkey path: NSEvent's global key
/// monitor and synthesizing the copy keystroke both require it.
enum Permissions {
    static func isAccessibilityTrusted() -> Bool {
        AXIsProcessTrusted()
    }

    /// Triggers the system "grant Accessibility access" dialog.
    @discardableResult
    static func promptForAccessibility() -> Bool {
        // Literal key = documented value of kAXTrustedCheckOptionPrompt; used
        // directly to avoid Unmanaged<CFString>-vs-CFString import differences.
        let options = ["AXTrustedCheckOptionPrompt": true] as CFDictionary
        return AXIsProcessTrustedWithOptions(options)
    }

    static func openAccessibilitySettings() {
        NSWorkspace.shared.open(URL(
            string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
        )!)
    }

    // MARK: - Screen Recording (Capture Region → ScreenCaptureKit)

    /// Screen Recording permission — required for `SCScreenshotManager` to return
    /// pixels. NOTE: unlike Accessibility, a running process keeps seeing the OLD
    /// value even after the user grants it; the app must be relaunched to capture.
    static func isScreenRecordingTrusted() -> Bool {
        CGPreflightScreenCaptureAccess()
    }

    /// Triggers the system "grant Screen Recording" dialog (first request only).
    @discardableResult
    static func requestScreenRecording() -> Bool {
        CGRequestScreenCaptureAccess()
    }

    static func openScreenRecordingSettings() {
        NSWorkspace.shared.open(URL(
            string: "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"
        )!)
    }
}
