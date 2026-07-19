import AppKit
import ApplicationServices

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
}
