# Audio File Organization

This directory contains all audio files for the Voice Conversion Comparison application.

## Directory Structure

```
audio/
├── demo/
│   ├── sources/          # Demo source audio files (5 files)
│   ├── targets/          # Demo target audio files (5 files)
│   └── results/          # Generated voice conversion results for demo mode
└── upload/
    └── results/          # Generated voice conversion results for upload mode
```

## File Naming Conventions
For basename naming convention, it should be {speaker_id}_{utterance_id}.

### Demo Mode Sources (`demo/sources/`)
- `female_speaker_1.wav`
- `male_speaker_1.wav`
- `female_speaker_2.wav`
- `male_speaker_2.wav`
- `child_speaker.wav`

### Demo Mode Targets (`demo/targets/`)
- `target_voice_a.wav`
- `target_voice_b.wav`
- `target_voice_c.wav`
- `target_voice_d.wav`
- `target_voice_e.wav`

### Generated Results

#### Demo Mode Results (`demo/results/`)
Format: `{model_name}/{target_basename}/{source_basename}.wav`

Examples:
- `DarkStream-v2-logits/target_voice_a/female_speaker_1.wav`
- `GenVC-small/target_voice_c/male_speaker_1.wav`
- `GenVC-large/target_voice_b/child_speaker.wav`

#### Upload Mode Results (`upload/results/`)
Format: `{model_name}_result.wav`

Examples:
- `DarkStream-v2-logits_result.wav`
- `GenVC-small_result.wav`

## Audio File Requirements

### Format Specifications
- **Format**: WAV (recommended) or MP3
- **Sample Rate**: 22kHz or 44.1kHz
- **Bit Depth**: 16-bit or 24-bit
- **Channels**: Mono (preferred) or Stereo
- **Duration**: 3-10 seconds for demo files

### Quality Guidelines
- Clear speech without background noise
- Consistent volume levels across all demo files
- Representative samples of different voice characteristics

## Adding New Audio Files

1. **Demo Sources**: Add to `demo/sources/` with descriptive names
2. **Demo Targets**: Add to `demo/targets/` with descriptive names
3. **Update Code**: Modify the arrays in `ComparisonView.vue`:
   - `demoSources` array
   - `demoTargets` array

## Backend Integration

When integrating with your voice conversion backend:

1. **API Endpoints**: Replace the mock functions in `ComparisonView.vue`:
   - `performVC()` for upload mode
   - `generateAllDemoVC()` for demo mode

2. **File Handling**: 
   - Upload mode: Send user files + model selection
   - Demo mode: Send source/target indices + model selection

3. **Result Storage**: Save generated files using the naming conventions above

## Security Considerations

- Validate file types and sizes on upload
- Implement rate limiting for generation requests
- Consider audio file cleanup policies for generated results
- Ensure proper file permissions and access controls