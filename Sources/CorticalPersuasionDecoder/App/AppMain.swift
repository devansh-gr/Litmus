import AppKit

/// Entry point. This is a background ("agent") app: no Dock icon, no window.
/// `.accessory` activation policy complements `LSUIElement` in Info.plist.
@main
struct CorticalPersuasionDecoderApp {
    static func main() {
        let app = NSApplication.shared
        let delegate = AppDelegate()
        // NSApplication.delegate is a weak reference; `delegate` stays alive
        // because this stack frame persists until `app.run()` returns at quit.
        app.delegate = delegate
        app.setActivationPolicy(.accessory)
        app.run()
    }
}
