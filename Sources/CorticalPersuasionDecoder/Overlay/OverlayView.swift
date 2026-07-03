import SwiftUI

/// The verdict card shown in the floating overlay: persuasion vector, the brain
/// region it targets, a confidence bar, and the mechanism.
struct VerdictCard: View {
    let title: String
    let brainRegion: String
    let mechanism: String
    let confidence: Double        // 0.0–1.0
    let rationale: String?

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
                Image(systemName: "waveform.badge.magnifyingglass")
                    .font(.caption2)
                Text("PERSUASION VECTOR")
                    .font(.caption2.weight(.semibold))
                    .tracking(1.2)
                Spacer(minLength: 8)
                Text("\(pct)%")
                    .font(.subheadline.weight(.bold))
                    .foregroundStyle(accent)
            }
            .foregroundStyle(.secondary)

            Text(title)
                .font(.title3.weight(.bold))

            HStack(spacing: 6) {
                Image(systemName: "brain.head.profile")
                Text("targets \(brainRegion)")
                    .font(.subheadline.weight(.medium))
            }
            .foregroundStyle(.primary.opacity(0.85))

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

            if let rationale, !rationale.isEmpty {
                Text(rationale)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(14)
        .frame(width: 320, alignment: .leading)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .strokeBorder(accent.opacity(0.35), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.25), radius: 14, y: 5)
    }
}
