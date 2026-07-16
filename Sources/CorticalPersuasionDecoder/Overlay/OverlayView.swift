import SwiftUI

/// The verdict card.
///
/// IMPORTANT — what this does and does NOT claim:
///   - The persuasion vector + confidence come from the LLM (the detector).
///   - The mechanism is a claim about the *psychological technique*, not anatomy.
///   - The brain regions are MEASURED by TRIBE v2 and z-scored against a baseline
///     corpus. We deliberately do NOT assert "fear -> amygdala": experiment A3
///     showed fear/outrage/reward all load on the SAME fronto-orbital regions
///     (r = 0.83-0.90), and this model cannot see the amygdala at all.
struct VerdictCard: View {
    let title: String
    let mechanism: String
    let confidence: Double          // 0.0-1.0
    let rationale: String?
    let profile: [CorticalSystem]?  // non-nil => brain map done
    let regionsFailed: Bool
    let awaitingBrain: Bool         // true => a deep scan is in flight (spinner)
                                    // false + nil profile => idle (show hint)

    private var pct: Int { Int((confidence * 100).rounded()) }

    private var accent: Color {
        switch confidence {
        case ..<0.34: return .secondary
        case ..<0.67: return .orange
        default:      return .red
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                Image(systemName: "waveform.badge.magnifyingglass").font(.caption2)
                Text("PERSUASION VECTOR")
                    .font(.caption2.weight(.semibold)).tracking(1.2)
                Spacer(minLength: 8)
                Text("\(pct)%")
                    .font(.subheadline.weight(.bold))
                    .foregroundStyle(accent)
            }
            .foregroundStyle(.secondary)

            Text(title).font(.title3.weight(.bold))

            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(.quaternary)
                    Capsule().fill(accent)
                        .frame(width: max(4, geo.size.width * confidence))
                }
            }
            .frame(height: 5)

            Text(mechanism)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            Divider().padding(.vertical, 2)

            HStack(spacing: 6) {
                Image(systemName: "brain.head.profile").font(.caption2)
                Text("CORTICAL IMPACT PROFILE")
                    .font(.caption2.weight(.semibold)).tracking(1.0)
            }
            .foregroundStyle(.secondary)

            brainSection
        }
        .padding(14)
        .frame(width: 340, alignment: .leading)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .strokeBorder(accent.opacity(0.35), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.25), radius: 14, y: 5)
    }

    @ViewBuilder
    private var brainSection: some View {
        if regionsFailed {
            Text("cortical map unavailable")
                .font(.caption2).foregroundStyle(.tertiary)
        } else if let profile {
            VStack(alignment: .leading, spacing: 5) {
                ForEach(profile, id: \.system) { s in
                    HStack(spacing: 8) {
                        Text(shortName(s.system))
                            .font(.caption)
                            .lineLimit(1)
                        Spacer(minLength: 6)
                        Text(s.level)
                            .font(.caption2.weight(.semibold))
                            .foregroundStyle(levelColor(s.z))
                    }
                    GeometryReader { geo in
                        ZStack(alignment: .leading) {
                            Capsule().fill(.quaternary)
                            Capsule().fill(levelColor(s.z))
                                // z ~ [-0.3, 1.0] mapped to bar width
                                .frame(width: max(3, geo.size.width * min(1, max(0, (s.z + 0.2) / 1.2))))
                        }
                    }
                    .frame(height: 4)
                }
                Text("TRIBE v2 · predicted engagement vs neutral text")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .padding(.top, 2)
            }
        } else if awaitingBrain {
            HStack(spacing: 6) {
                ProgressView().controlSize(.small)
                Text("modelling cortex… (~2 min)")
                    .font(.caption2).foregroundStyle(.tertiary)
            }
        } else {
            // Idle: the brain map is opt-in (it takes ~2 min), so don't auto-run it.
            HStack(spacing: 5) {
                Image(systemName: "brain.head.profile").font(.caption2)
                Text("Deep-scan cortex →  🧠 menu")
                    .font(.caption2)
            }
            .foregroundStyle(.tertiary)
        }
    }

    /// Trim the anatomical parenthetical for a compact card row, e.g.
    /// "Value / evaluation (orbitofrontal)" -> "Value / evaluation".
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
