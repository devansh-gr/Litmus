import ScreenCaptureKit
import AppKit
import CoreGraphics

enum RegionCaptureError: Error, LocalizedError {
    case noDisplay
    case cropFailed
    var errorDescription: String? {
        switch self {
        case .noDisplay: return "Could not find a display for the selection."
        case .cropFailed: return "Failed to crop the captured image."
        }
    }
}

/// Captures a screen region as a CGImage via ScreenCaptureKit.
///
/// Strategy: screenshot the whole display at native resolution, then crop the
/// CGImage to the selection in pixel space. Cropping a CGImage is unambiguous
/// (top-left pixel origin), which avoids the fiddly `sourceRect` coordinate
/// conventions. `globalRect` is in AppKit screen coordinates (bottom-left origin,
/// points).
enum RegionCapture {

    static func capture(globalRect: CGRect) async throws -> CGImage {
        guard let screen = NSScreen.screens.first(where: { $0.frame.intersects(globalRect) })
                ?? NSScreen.main else {
            throw RegionCaptureError.noDisplay
        }
        let scale = screen.backingScaleFactor
        let displayID = screen.displayID

        let content = try await SCShareableContent.current
        guard let scDisplay = content.displays.first(where: { $0.displayID == displayID })
                ?? content.displays.first else {
            throw RegionCaptureError.noDisplay
        }

        // Full-display screenshot at native resolution.
        let filter = SCContentFilter(display: scDisplay, excludingWindows: [])
        let config = SCStreamConfiguration()
        config.width = Int(CGFloat(scDisplay.width) * scale)
        config.height = Int(CGFloat(scDisplay.height) * scale)
        config.showsCursor = false
        let fullImage = try await SCScreenshotManager.captureImage(contentFilter: filter, configuration: config)

        // Global (bottom-left) points → display-local top-left pixels.
        let localX = globalRect.minX - screen.frame.minX
        let localBottomY = globalRect.minY - screen.frame.minY
        let localTopY = screen.frame.height - (localBottomY + globalRect.height)
        let pixelRect = CGRect(x: localX * scale,
                               y: localTopY * scale,
                               width: globalRect.width * scale,
                               height: globalRect.height * scale).integral

        guard let cropped = fullImage.cropping(to: pixelRect) else {
            throw RegionCaptureError.cropFailed
        }
        return cropped
    }
}

extension NSScreen {
    var displayID: CGDirectDisplayID {
        (deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? NSNumber)?.uint32Value ?? 0
    }
}
