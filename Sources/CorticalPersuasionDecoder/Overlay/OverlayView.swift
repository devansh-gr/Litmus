import SwiftUI

/// A small transient notice ("Analyzing…", "No manipulation detected ✓", errors),
/// shown in the same floating panel style as the verdict card.
struct NoticeCard: View {
    let message: String

    private var kind: (icon: String, color: Color) {
        let m = message.lowercased()
        if message.contains("✓")            { return ("checkmark.seal.fill", .green) }
        if m.contains("analy")              { return ("sparkles", .blue) }
        if m.contains("warm")               { return ("hourglass", .orange) }
        if m.contains("offline") || m.contains("error") { return ("exclamationmark.triangle.fill", .orange) }
        return ("info.circle.fill", .secondary)
    }

    var body: some View {
        HStack(alignment: .center, spacing: 9) {
            Image(systemName: kind.icon)
                .foregroundStyle(kind.color)
                .font(.body)
                .symbolEffect(.pulse, options: .repeating, isActive: message.lowercased().contains("analy"))
            Text(message)
                .font(.callout)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 14).padding(.vertical, 11)
        .frame(width: 290, alignment: .leading)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 13, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 13, style: .continuous)
            .strokeBorder(kind.color.opacity(0.35), lineWidth: 1))
        .shadow(color: .black.opacity(0.28), radius: 14, y: 5)
    }
}

/// A runner-up vector in the mixture line.
struct MixtureItem {
    let label: String
    let pct: Int
}

// MARK: - Mini brain

/// A stylized LATERAL cortex (frontal lobe at the left) that glows over the
/// frontal systems the content engages. Intensities come from the MEASURED impact
/// profile when a deep-scan has run; before that it shows the *typical* persuasion
/// cortex (value + language) at a fixed low glow — a general fact about persuasive
/// language (experiment A7), NOT a per-vector anatomical claim. Positions are
/// approximate on purpose ("roughly is okay").
struct MiniBrainView: View {
    let profile: [CorticalSystem]?

    private struct Region { let key: String; let at: CGPoint; let typical: Double }
    private static let regions: [Region] = [
        .init(key: "dlpfc",            at: CGPoint(x: 0.35, y: 0.30), typical: 0.20),  // executive
        .init(key: "inferior frontal", at: CGPoint(x: 0.25, y: 0.48), typical: 0.55),  // language
        .init(key: "orbitofrontal",    at: CGPoint(x: 0.28, y: 0.63), typical: 0.62),  // value
    ]

    private var measured: Bool { profile != nil }

    private func intensity(_ r: Region) -> Double {
        if let profile, let s = profile.first(where: { $0.system.lowercased().contains(r.key) }) {
            return min(1, max(0, (s.z + 0.2) / 1.2))
        }
        return measured ? 0 : r.typical
    }

    private func heat(_ t: Double) -> Color {
        Color(hue: 0.62 - 0.62 * min(1, max(0, t)), saturation: 0.85, brightness: 0.98)
    }

    var body: some View {
        Canvas { ctx, size in
            let brain = Self.brainPath(size)

            ctx.fill(brain, with: .color(.primary.opacity(0.06)))

            // Region glows, clipped to the cortex silhouette.
            ctx.drawLayer { layer in
                layer.clip(to: brain)
                for r in Self.regions {
                    let t = intensity(r)
                    guard t > 0.03 else { continue }
                    let c = heat(t)
                    let center = CGPoint(x: r.at.x * size.width, y: r.at.y * size.height)
                    let rad = min(size.width, size.height) * (0.20 + 0.18 * t)
                    let rect = CGRect(x: center.x - rad, y: center.y - rad, width: rad * 2, height: rad * 2)
                    layer.fill(Path(ellipseIn: rect),
                               with: .radialGradient(Gradient(colors: [c.opacity(0.9), c.opacity(0)]),
                                                     center: center, startRadius: 0, endRadius: rad))
                }
            }

            // Sulci texture + outline on top.
            for arc in Self.sulci(size) {
                ctx.stroke(arc, with: .color(.secondary.opacity(0.28)), lineWidth: 0.8)
            }
            ctx.stroke(brain, with: .color(.secondary.opacity(0.55)),
                       style: StrokeStyle(lineWidth: 1.3, lineJoin: .round))

            // Tiny locator dots so cool regions are still readable.
            for r in Self.regions {
                let center = CGPoint(x: r.at.x * size.width, y: r.at.y * size.height)
                let d = 2.4
                ctx.fill(Path(ellipseIn: CGRect(x: center.x - d, y: center.y - d, width: d * 2, height: d * 2)),
                         with: .color(.primary.opacity(0.28)))
            }
        }
        .accessibilityHidden(true)
    }

    /// Left-facing lateral cerebrum + cerebellum + brainstem, in `size` coords.
    private static func brainPath(_ size: CGSize) -> Path {
        let w = size.width, h = size.height
        let P = { (x: CGFloat, y: CGFloat) in CGPoint(x: x * w, y: y * h) }
        var p = Path()
        p.move(to: P(0.12, 0.52))
        p.addCurve(to: P(0.32, 0.14), control1: P(0.12, 0.28), control2: P(0.18, 0.16))
        p.addCurve(to: P(0.74, 0.14), control1: P(0.46, 0.05), control2: P(0.62, 0.05))
        p.addCurve(to: P(0.90, 0.42), control1: P(0.86, 0.18), control2: P(0.93, 0.30))
        p.addCurve(to: P(0.70, 0.64), control1: P(0.89, 0.56), control2: P(0.82, 0.63))
        p.addCurve(to: P(0.30, 0.66), control1: P(0.58, 0.66), control2: P(0.42, 0.69))
        p.addCurve(to: P(0.12, 0.52), control1: P(0.20, 0.65), control2: P(0.11, 0.60))
        p.closeSubpath()
        p.addEllipse(in: CGRect(x: 0.66 * w, y: 0.56 * h, width: 0.21 * w, height: 0.22 * h))  // cerebellum
        p.addPath(Path(roundedRect: CGRect(x: 0.44 * w, y: 0.64 * h, width: 0.10 * w, height: 0.20 * h),
                       cornerRadius: 0.03 * w))                                                  // brainstem
        return p
    }

    private static func sulci(_ size: CGSize) -> [Path] {
        let w = size.width, h = size.height
        let P = { (x: CGFloat, y: CGFloat) in CGPoint(x: x * w, y: y * h) }
        func arc(_ m: CGPoint, _ c1: CGPoint, _ c2: CGPoint, _ e: CGPoint) -> Path {
            var a = Path(); a.move(to: m); a.addCurve(to: e, control1: c1, control2: c2); return a
        }
        return [
            arc(P(0.31, 0.22), P(0.24, 0.34), P(0.40, 0.40), P(0.35, 0.54)),
            arc(P(0.50, 0.15), P(0.44, 0.36), P(0.60, 0.42), P(0.53, 0.60)),
            arc(P(0.68, 0.18), P(0.62, 0.36), P(0.78, 0.42), P(0.67, 0.58)),
        ]
    }
}

// MARK: - Verdict card

/// The verdict card.
///
/// IMPORTANT — what this does and does NOT claim:
///   - The persuasion vector + confidence come from the LLM (the detector).
///   - The mechanism is a claim about the *psychological technique*, not anatomy.
///   - The brain regions are MEASURED by TRIBE v2 and z-scored against a baseline.
///     We deliberately do NOT assert "fear -> amygdala": experiment A3 showed
///     fear/outrage/reward all load on the SAME fronto-orbital regions, and this
///     model cannot see the amygdala at all.
struct VerdictCard: View {
    let title: String
    let mechanism: String
    let confidence: Double          // 0.0-1.0
    let rationale: String?
    let mixture: [MixtureItem]
    let profile: [CorticalSystem]?  // non-nil => brain map done
    let regionsFailed: Bool
    let awaitingBrain: Bool

    private var pct: Int { Int((confidence * 100).rounded()) }

    private var accent: Color {
        switch confidence {
        case ..<0.34: return .secondary
        case ..<0.67: return .orange
        default:      return .red
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 11) {
            header
            Text(title).font(.title3.weight(.bold)).fixedSize(horizontal: false, vertical: true)
            confidenceBar

            Text(mechanism)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            if !mixture.isEmpty {
                Text("also: " + mixture.map { "\($0.label) \($0.pct)%" }.joined(separator: " · "))
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Divider().padding(.vertical, 1)

            HStack(spacing: 6) {
                Image(systemName: "brain.head.profile").font(.caption2)
                Text("WHERE IT LANDS IN CORTEX")
                    .font(.caption2.weight(.semibold)).tracking(0.9)
            }
            .foregroundStyle(.secondary)

            brainSection
        }
        .padding(16)
        .frame(width: 360, alignment: .leading)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(
                    LinearGradient(colors: [accent.opacity(0.55), accent.opacity(0.15)],
                                   startPoint: .topLeading, endPoint: .bottomTrailing),
                    lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.3), radius: 18, y: 6)
    }

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "waveform.badge.magnifyingglass")
                .font(.footnote.weight(.semibold))
                .foregroundStyle(accent)
                .frame(width: 26, height: 26)
                .background(accent.opacity(0.15), in: Circle())
            Text("PERSUASION VECTOR")
                .font(.caption2.weight(.semibold)).tracking(1.3)
                .foregroundStyle(.secondary)
            Spacer(minLength: 8)
            Text("\(pct)%")
                .font(.title3.weight(.heavy))
                .foregroundStyle(accent)
                .contentTransition(.numericText())
        }
    }

    private var confidenceBar: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule().fill(.quaternary)
                Capsule()
                    .fill(LinearGradient(colors: [accent.opacity(0.7), accent],
                                         startPoint: .leading, endPoint: .trailing))
                    .frame(width: max(5, geo.size.width * confidence))
            }
        }
        .frame(height: 6)
    }

    @ViewBuilder
    private var brainSection: some View {
        HStack(alignment: .center, spacing: 14) {
            MiniBrainView(profile: regionsFailed ? nil : profile)
                .frame(width: 128, height: 100)
                .opacity(regionsFailed ? 0.35 : 1)

            VStack(alignment: .leading, spacing: 6) {
                if regionsFailed {
                    Label("cortical map unavailable", systemImage: "xmark.circle")
                        .font(.caption2).foregroundStyle(.tertiary)
                } else if let profile {
                    ForEach(profile, id: \.system) { s in
                        HStack(spacing: 7) {
                            Circle().fill(levelColor(s.z)).frame(width: 7, height: 7)
                            Text(shortName(s.system)).font(.caption).lineLimit(1)
                            Spacer(minLength: 4)
                            Text(s.level).font(.caption2.weight(.semibold))
                                .foregroundStyle(levelColor(s.z))
                        }
                    }
                    Text("measured vs neutral · TRIBE v2")
                        .font(.system(size: 9)).foregroundStyle(.tertiary).padding(.top, 1)
                } else if awaitingBrain {
                    HStack(spacing: 6) {
                        ProgressView().controlSize(.small)
                        Text("modelling cortex…").font(.caption2).foregroundStyle(.tertiary)
                    }
                } else {
                    Text("Rough map — value + language cortex.")
                        .font(.caption2).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    Label("Deep-scan (🧠) to measure this text", systemImage: "scope")
                        .font(.system(size: 9)).foregroundStyle(.tertiary)
                }
            }
        }
    }

    private func shortName(_ system: String) -> String {
        if let paren = system.firstIndex(of: "(") {
            return String(system[..<paren]).trimmingCharacters(in: .whitespaces)
        }
        return system
    }

    private func levelColor(_ z: Double) -> Color {
        switch z {
        case ..<(-0.15): return .blue
        case ..<0.15:    return .secondary
        case ..<0.5:     return .orange
        default:         return .red
        }
    }
}
