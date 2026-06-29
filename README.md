# Cortical Persuasion Decoder

A real-time "cognitive-manipulation x-ray" for macOS. Highlight text or select a
screen region and it labels the **persuasion vector** the content uses, the
**brain region** that vector targets, and a **confidence score** — rendered in a
floating overlay beside your selection.

> **Status: Milestone 1 — capture spike.**
> Right now the app only proves the riskiest unknown: reading your live text
> selection via the Accessibility API and clearing the macOS permission gate. It
> prints each selection to the console. Region capture + OCR, the classifier,
> and the overlay arrive in later milestones.

## Requirements

- macOS 13 or later
- **Full Xcode** (Command Line Tools alone are not enough) + an Apple ID for code signing
- [XcodeGen](https://github.com/yonaskolb/XcodeGen) — `brew install xcodegen` — generates the Xcode project from `project.yml`

## Permissions

| Permission | Needed for | Introduced in |
|---|---|---|
| **Accessibility** | Reading the currently selected text (`kAXSelectedText`) | Milestone 1 |
| **Screen Recording** | Region capture + OCR | Milestone 2 |

Grant Accessibility under **System Settings ▸ Privacy & Security ▸ Accessibility**
and toggle **CorticalPersuasionDecoder** on.

## Build & run (Milestone 1)

```sh
brew install xcodegen        # one-time
xcodegen generate            # creates CorticalPersuasionDecoder.xcodeproj from project.yml
open CorticalPersuasionDecoder.xcodeproj
```

In Xcode: select the **CorticalPersuasionDecoder** target ▸ **Signing &
Capabilities** ▸ choose your **Team**, then **Run (⌘R)**.

This is a background agent app (`LSUIElement`) — no Dock icon, no window. To see
selections print, use any of:

- **Xcode console** — output appears there when you Run.
- **Terminal** — run the built binary directly:
  ```sh
  APP=$(ls -dt ~/Library/Developer/Xcode/DerivedData/CorticalPersuasionDecoder-*/Build/Products/Debug/CorticalPersuasionDecoder.app | head -1)
  "$APP/Contents/MacOS/CorticalPersuasionDecoder"
  ```
- **Console.app** — filter by subsystem `ai.zeonsystems.corticalpersuasiondecoder`.

On first launch, grant Accessibility (above), then highlight text in
**TextEdit / Safari / Notes** and watch it print. Chrome/Electron web content
won't report yet — that needs the ⌘C fallback added in a later milestone.

## Configuration (later milestones)

The remote classifier will be configured via environment variables
(`CPD_ENDPOINT_URL`, `CPD_MODEL_NAME`, `CPD_API_KEY`, `CPD_CLASSIFIER=mock|remote`)
and/or `~/.config/cortical-persuasion-decoder/config.json`.

**No telemetry:** screen content leaves the machine only on a remote-classifier
call to *your* configured endpoint. The mock classifier is fully offline.
