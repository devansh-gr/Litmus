import AppKit
import ApplicationServices

/// Accessibility permission flow for the capture pipeline.
enum Permissions {

    /// Whether this app is currently trusted to use the Accessibility API.
    static func isAccessibilityTrusted() -> Bool {
        AXIsProcessTrusted()
    }

    /// Triggers the system "grant Accessibility access" dialog and returns the
    /// current trust state. Safe to call when already trusted (no dialog shown).
    @discardableResult
    static func promptForAccessibility() -> Bool {
        // Literal key string is the documented value of
        // `kAXTrustedCheckOptionPrompt`; using it directly avoids the
        // Unmanaged<CFString>-vs-CFString import differences across SDKs.
        let options = ["AXTrustedCheckOptionPrompt": true] as CFDictionary
        return AXIsProcessTrustedWithOptions(options)
    }

    /// Deep link to System Settings ▸ Privacy & Security ▸ Accessibility.
    static let accessibilitySettingsURL = URL(
        string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
    )!

    static func openAccessibilitySettings() {
        NSWorkspace.shared.open(accessibilitySettingsURL)
    }
}
