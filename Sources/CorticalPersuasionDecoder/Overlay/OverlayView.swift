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
    let regions: [BrainRegion]?     // nil => still computing
    let regionsFailed: Bool

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
                Text("PREDICTED CORTICAL ACTIVATION")
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
        } else if let regions {
            VStack(alignment: .leading, spacing: 4) {
                ForEach(regions.prefix(4), id: \.region) { r in
                    HStack(spacing: 8) {
                        Text(pretty(r.region))
                            .font(.caption)
                            .lineLimit(1)
                        Spacer(minLength: 6)
                        Text(String(format: "z%+.1f", r.activationZ))
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(r.activationZ >= 0 ? .red : .blue)
                    }
                }
                Text("TRIBE v2 · measured, not asserted")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .padding(.top, 2)
            }
        } else {
            HStack(spacing: 6) {
                ProgressView().controlSize(.small)
                Text("modelling cortex…")
                    .font(.caption2).foregroundStyle(.tertiary)
            }
        }
    }

    /// Destrieux labels are unreadable ("G_front_inf-Triangul"); humanise them.
    private func pretty(_ raw: String) -> String {
        let map: [String: String] = [
            "G_front_inf-Triangul": "Inferior frontal (Broca's)",
            "G_front_inf-Orbital": "Inferior frontal, orbital",
            "G_front_inf-Opercular": "Inferior frontal, opercular",
            "G_front_middle": "Middle frontal (dlPFC)",
            "G_front_sup": "Superior frontal",
            "S_orbital-H_Shaped": "Orbital sulcus",
            "S_orbital_med-olfact": "Medial orbital sulcus",
            "G_rectus": "Gyrus rectus (vmPFC)",
            "G_and_S_cingul-Ant": "Anterior cingulate",
            "G_Ins_lg_and_S_cent_ins": "Insula",
            "G_temp_sup-Plan_polar": "Superior temporal (planum polare)",
            "S_temporal_sup": "Superior temporal sulcus",
            "S_intrapariet_and_P_trans": "Intraparietal sulcus",
        ]
        if let nice = map[raw] { return nice }
        return raw.replacingOccurrences(of: "_", with: " ")
    }
}
