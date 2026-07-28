// Headless proof that the OCR half of Capture Region works — no Screen Recording
// permission needed. Renders known text to an image, then runs the EXACT Vision
// request config from Sources/…/Capture/OCRService.swift and checks the round-trip.
//
//   swift scripts/ocr_selftest.swift
//
// If this prints PASS, OCRService is sound and any Capture Region failure is upstream
// (the ScreenCaptureKit screenshot / Screen Recording permission), not the OCR step.

import AppKit
import Vision

func renderText(_ text: String, width: Int, height: Int) -> CGImage? {
    guard let rep = NSBitmapImageRep(
        bitmapDataPlanes: nil, pixelsWide: width, pixelsHigh: height,
        bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
        colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0
    ) else { return nil }

    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
    NSColor.white.setFill()
    NSRect(x: 0, y: 0, width: width, height: height).fill()
    let attrs: [NSAttributedString.Key: Any] = [
        .font: NSFont.systemFont(ofSize: 44, weight: .semibold),
        .foregroundColor: NSColor.black,
    ]
    (text as NSString).draw(at: NSPoint(x: 28, y: Double(height) / 2 - 26), withAttributes: attrs)
    NSGraphicsContext.restoreGraphicsState()
    return rep.cgImage
}

// Same config as OCRService.recognizeText(in:).
func recognizeText(in image: CGImage) throws -> String {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    let handler = VNImageRequestHandler(cgImage: image, options: [:])
    try handler.perform([request])
    let observations = request.results ?? []
    return observations
        .compactMap { $0.topCandidates(1).first?.string }
        .joined(separator: "\n")
}

let sample = "URGENT: act now before it is too late"
guard let cg = renderText(sample, width: 920, height: 150) else {
    print("FAIL — could not render the test image"); exit(1)
}
do {
    let out = try recognizeText(in: cg)
    print("input : \(sample)")
    print("ocr   : \(out.replacingOccurrences(of: "\n", with: " / "))")
    if out.lowercased().contains("act now") {
        print("PASS \u{2713} — OCR pipeline works; any Capture Region failure is upstream (capture/permission).")
        exit(0)
    } else {
        print("CHECK — OCR ran but output didn't match; inspect the config.")
        exit(2)
    }
} catch {
    print("FAIL — Vision threw: \(error.localizedDescription)")
    exit(1)
}
