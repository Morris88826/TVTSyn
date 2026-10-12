<template>
  <div class="page">
    <!-- Hero -->
    <header class="hero">
      <div class="wrap">
        <h1 class="title">
          <span class="title-name">TVTSyn</span>: Content-Synchronous Time-Varying Timbre for
          Streaming Voice Conversion and Anonymization
        </h1>

        <div class="authors">
          <span v-for="author in authors" :key="author.name" class="author">
            <a :href="author.url" target="_blank" rel="noopener noreferrer">{{ author.name }}</a
            ><sup>1</sup>
          </span>
        </div>
        <div class="affiliations"><sup>1</sup>Texas A&amp;M University</div>

        <div class="venue">
          <span class="venue-name"
            >The Fourteenth International Conference on Learning Representations (ICLR 2026)</span
          >
        </div>

        <nav class="links">
          <a
            v-for="link in links"
            :key="link.label"
            :href="link.href"
            :target="link.external ? '_blank' : undefined"
            :rel="link.external ? 'noopener noreferrer' : undefined"
            class="link-btn"
          >
            <img v-if="link.image" :src="getAssetUrl(link.image)" alt="" />
            <svg v-else viewBox="0 0 16 16" aria-hidden="true"><path :d="link.icon" /></svg>
            <span>{{ link.label }}</span>
          </a>
        </nav>
      </div>
    </header>

    <main>
      <!-- Abstract -->
      <section class="section">
        <div class="wrap">
          <h2 class="section-title">Abstract</h2>
          <p class="abstract">
            Real-time voice conversion and speaker anonymization require causal, low-latency
            synthesis without sacrificing intelligibility or naturalness. Current systems have a
            core representational mismatch: content is time-varying, while speaker identity is
            injected as a static global embedding. We introduce a streamable speech synthesizer that
            aligns the temporal granularity of identity and content via a content-synchronous,
            time-varying timbre (TVT) representation. A Global Timbre Memory expands a global timbre
            instance into multiple compact facets; frame-level content attends to this memory, a
            gate regulates variation, and spherical interpolation preserves identity geometry while
            enabling smooth local changes. In addition, a factorized vector-quantized bottleneck
            regularizes content to reduce residual speaker leakage. The resulting system is
            streamable end-to-end, with &lt;80 ms GPU latency. Experiments show improvements in
            naturalness, speaker transfer, and anonymization compared to SOTA streaming baselines,
            establishing TVT as a scalable approach for privacy-preserving and expressive speech
            synthesis under strict latency budgets.
          </p>
        </div>
      </section>

      <!-- Method -->
      <section class="section section-alt">
        <div class="wrap wrap-wide">
          <h2 class="section-title">Method</h2>
          <figure class="figure">
            <img
              :src="getAssetUrl('/architectures/architecture.png')"
              alt="TVTSyn model architecture"
              loading="lazy"
            />
            <figcaption>
              <b>Figure 1.</b> Overview of the <u>content-synchronous time-varying timbre (TVT)</u>
              framework. A Global Timbre Memory expands a global timbre instance into compact
              facets; frame-level content attends to this memory, a gate regulates variation, and
              spherical interpolation produces a smoothly varying, identity-preserving timbre. A
              factorized vector-quantized bottleneck regularizes content to reduce residual speaker
              leakage.
            </figcaption>
          </figure>
        </div>
      </section>

      <!-- Audio Samples -->
      <section class="section" id="samples">
        <div class="wrap wrap-wide">
          <h2 class="section-title">Audio Samples</h2>
          <p class="section-lead">
            Any-to-any voice conversion under streaming constraints. Select a source utterance, a
            target speaker, and the systems to compare.
          </p>

          <div class="toolbar">
            <div class="toolbar-group">
              <span class="toolbar-label">Source</span>
              <div class="stepper">
                <button
                  class="step-btn"
                  :disabled="sourceIndex === 0"
                  aria-label="Previous source"
                  @click="sourceIndex--"
                >
                  ‹
                </button>
                <select v-model="sourceIndex" class="example-select" aria-label="Source">
                  <option v-for="(source, idx) in sources" :key="source" :value="idx">
                    {{ formatIndex(idx) }} / {{ formatIndex(sources.length - 1) }}
                  </option>
                </select>
                <button
                  class="step-btn"
                  :disabled="sourceIndex === sources.length - 1"
                  aria-label="Next source"
                  @click="sourceIndex++"
                >
                  ›
                </button>
              </div>
            </div>

            <div class="toolbar-group">
              <span class="toolbar-label">Target</span>
              <div class="stepper">
                <button
                  class="step-btn"
                  :disabled="targetIndex === 0"
                  aria-label="Previous target"
                  @click="targetIndex--"
                >
                  ‹
                </button>
                <select v-model="targetIndex" class="example-select" aria-label="Target">
                  <option v-for="(target, idx) in targets" :key="target" :value="idx">
                    {{ formatIndex(idx) }} / {{ formatIndex(targets.length - 1) }}
                  </option>
                </select>
                <button
                  class="step-btn"
                  :disabled="targetIndex === targets.length - 1"
                  aria-label="Next target"
                  @click="targetIndex++"
                >
                  ›
                </button>
              </div>
            </div>

            <div class="toolbar-group">
              <span class="toolbar-label">Systems</span>
              <div class="chips">
                <button
                  v-for="model in availableModels"
                  :key="model"
                  class="chip"
                  :class="{ active: selectedModels.includes(model), ours: model === 'TVTSyn' }"
                  :aria-pressed="selectedModels.includes(model)"
                  @click="toggleModel(model)"
                >
                  {{ model }}
                </button>
              </div>
            </div>
          </div>

          <div class="sample-group">
            <div class="group-label">Reference</div>
            <div class="audio-grid">
              <div class="audio-card">
                <div class="audio-name">Source</div>
                <div class="audio-meta">{{ sources[sourceIndex] }}</div>
                <audio
                  :src="getAssetUrl(`/audio/demo/sources/${sources[sourceIndex]}.wav`)"
                  controls
                  preload="none"
                ></audio>
              </div>
              <div class="audio-card">
                <div class="audio-name">Target speaker</div>
                <div class="audio-meta">{{ targets[targetIndex] }}</div>
                <audio
                  :src="getAssetUrl(`/audio/demo/targets/${targets[targetIndex]}.wav`)"
                  controls
                  preload="none"
                ></audio>
              </div>
            </div>
          </div>

          <div class="sample-group">
            <div class="group-label">Converted</div>
            <div class="audio-grid">
              <div
                v-for="model in selectedModels"
                :key="model"
                class="audio-card"
                :class="{ 'audio-card-ours': model === 'TVTSyn' }"
              >
                <div class="audio-name">
                  {{ model }}
                  <span v-if="model === 'TVTSyn'" class="ours-tag">Ours</span>
                </div>
                <audio
                  :src="
                    getAssetUrl(
                      `/audio/demo/results/${model}/${targets[targetIndex]}/${sources[sourceIndex]}.wav`,
                    )
                  "
                  controls
                  preload="none"
                ></audio>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- BibTeX -->
      <section class="section section-alt" id="bibtex">
        <div class="wrap">
          <div class="bibtex-header">
            <h2 class="section-title">BibTeX</h2>
            <button class="copy-btn" @click="copyBibtex">
              {{ copied ? 'Copied' : 'Copy' }}
            </button>
          </div>
          <pre class="bibtex"><code v-text="bibtex"></code></pre>
        </div>
      </section>
    </main>

    <footer class="footer">
      <div class="wrap">
        © 2026 Texas A&amp;M University ·
        <a href="https://github.com/PSI-TAMU/TVTSyn" target="_blank" rel="noopener noreferrer"
          >GitHub</a
        >
      </div>
    </footer>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const getAssetUrl = (path) => {
  const base = import.meta.env.BASE_URL || '/'
  const cleanPath = path.startsWith('/') ? path.slice(1) : path
  const cleanBase = base.endsWith('/') ? base : base + '/'
  return cleanBase + cleanPath
}

const authors = [
  { name: 'Waris Quamer', url: 'https://github.com/warisqr007' },
  { name: 'Mu-Ruei Tseng', url: 'https://github.com/Morris88826' },
  { name: 'Ghady Nasrallah', url: 'https://github.com/Ghadynasrallah' },
  {
    name: 'Ricardo Gutierrez-Osuna',
    url: 'https://scholar.google.com/citations?user=UnuQfEwAAAAJ&hl=en',
  },
]

const links = [
  {
    label: 'Paper',
    href: 'https://arxiv.org/abs/2602.09389',
    external: true,
    icon: 'M4 0h5.5L14 4.5V14a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V2a2 2 0 0 1 2-2zm5 1.5V5h3.5L9 1.5zM5 8.5a.5.5 0 0 0 0 1h6a.5.5 0 0 0 0-1H5zm0 2a.5.5 0 0 0 0 1h6a.5.5 0 0 0 0-1H5zm0 2a.5.5 0 0 0 0 1h4a.5.5 0 0 0 0-1H5z',
  },
  {
    label: 'Code',
    href: 'https://github.com/PSI-TAMU/TVTSyn',
    external: true,
    icon: 'M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z',
  },
  {
    label: 'Hugging Face',
    href: 'https://huggingface.co/warisqr007/TVTSyn',
    external: true,
    image: '/icons/huggingface.svg',
  },
  {
    label: 'Samples',
    href: '#samples',
    icon: 'M7 2.5a.5.5 0 0 0-.8-.4L3.3 4.5H1.5A.5.5 0 0 0 1 5v6a.5.5 0 0 0 .5.5h1.8l2.9 2.4A.5.5 0 0 0 7 13.5v-11zm3.5 1.7a.5.5 0 0 1 .7 0A5.5 5.5 0 0 1 11.2 12a.5.5 0 1 1-.7-.7 4.5 4.5 0 0 0 0-6.4.5.5 0 0 1 0-.7zM9.1 5.6a.5.5 0 0 1 .7 0 3.5 3.5 0 0 1 0 4.9.5.5 0 0 1-.7-.7 2.5 2.5 0 0 0 0-3.5.5.5 0 0 1 0-.7z',
  },
  {
    label: 'BibTeX',
    href: '#bibtex',
    icon: 'M3.5 1A1.5 1.5 0 0 0 2 2.5v11a.5.5 0 0 0 .8.4L8 10.1l5.2 3.8a.5.5 0 0 0 .8-.4v-11A1.5 1.5 0 0 0 12.5 1h-9z',
  },
]

const availableModels = ['slt24', 'DarkStream', 'GenVC-small', 'TVTSyn-novq', 'TVTSyn']

const selectedModels = ref(['DarkStream', 'GenVC-small', 'TVTSyn'])

const toggleModel = (model) => {
  const isSelected = selectedModels.value.includes(model)
  if (isSelected && selectedModels.value.length === 1) return
  const next = isSelected
    ? selectedModels.value.filter((m) => m !== model)
    : [...selectedModels.value, model]
  // Keep the display order consistent with availableModels.
  selectedModels.value = availableModels.filter((m) => next.includes(m))
}

const sources = [
  'BDL_arctic_b0290',
  'BDL_arctic_b0509',
  'BWC_arctic_b0191',
  'NJS_arctic_b0348',
  'p232_p232_194',
  'p252_p252_165',
  'p252_p252_350',
  'p265_p265_122',
  'p271_p271_007',
  'p276_p276_365',
  'p336_p336_112',
  'p374_p374_039',
  'RMS_arctic_a0125',
  'SLT_arctic_a0410',
  'YDCK_arctic_a0261',
]

const targets = [
  'EF2_ENG_0024_0',
  'EF2_ENG_0115_1',
  'EF4_ENG_0003_1',
  'EF4_ENG_0044_0',
  'EF6_ENG_0006_1',
  'EF7_ENG_0048_0',
  'EM1_ENG_0091_0',
  'EM4_ENG_0001_1',
  'EM4_ENG_0087_1',
  'EM4_ENG_0089_0',
  'EM5_ENG_0009_0',
  'EM6_ENG_0076_0',
]

const sourceIndex = ref(0)
const targetIndex = ref(0)

const formatIndex = (idx) => String(idx + 1).padStart(2, '0')

const bibtex = `@inproceedings{quamer2026tvtsyn,
  title={{TVTSyn}: Content-Synchronous Time-Varying Timbre for Streaming Voice Conversion and Anonymization},
  author={Waris Quamer and Mu-Ruei Tseng and Ghady Nasrallah and Ricardo Gutierrez-Osuna},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=Tf4Lfw85lS}
}`

const copied = ref(false)

const copyBibtex = async () => {
  try {
    await navigator.clipboard.writeText(bibtex)
    copied.value = true
    setTimeout(() => (copied.value = false), 2000)
  } catch {
    // Clipboard unavailable (e.g. non-HTTPS); text remains selectable.
  }
}
</script>

<style scoped>
.page {
  --ink: #1a1a1a;
  --muted: #5f6368;
  --faint: #9aa0a6;
  --line: #e6e6e6;
  --alt: #fafafa;
  --link: #1a5fb4;
  --accent: #b5121b;

  min-height: 100vh;
  background: #ffffff;
  color: var(--ink);
  font-family:
    'Noto Sans',
    -apple-system,
    BlinkMacSystemFont,
    'Segoe UI',
    sans-serif;
  font-size: 16px;
  line-height: 1.6;
}

.wrap {
  max-width: 860px;
  margin: 0 auto;
  padding: 0 20px;
}

.wrap-wide {
  max-width: 1040px;
}

a {
  color: var(--link);
  text-decoration: none;
}

a:hover {
  text-decoration: underline;
}

/* Hero */
.hero {
  padding: 72px 0 48px;
  text-align: center;
}

.title {
  font-family: 'Google Sans', 'Noto Sans', sans-serif;
  font-size: 40px;
  font-weight: 600;
  line-height: 1.25;
  letter-spacing: -0.5px;
  margin: 0 auto 24px;
  max-width: 820px;
}

.title-name {
  font-weight: 700;
}

.authors {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 4px 22px;
  font-size: 18px;
}

.author {
  white-space: nowrap;
}

.authors sup,
.affiliations sup {
  font-size: 0.65em;
  margin-left: 1px;
  color: var(--muted);
}

.affiliations {
  margin-top: 6px;
  font-size: 16px;
  color: var(--muted);
}

.venue {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  align-items: center;
  gap: 10px;
  margin-top: 18px;
}

.venue-name {
  font-size: 17px;
  font-weight: 600;
}

.links {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 10px;
  margin-top: 28px;
}

.link-btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 20px;
  border-radius: 999px;
  background: #363636;
  color: #ffffff;
  font-size: 15px;
  font-weight: 500;
  transition: background 0.15s ease;
}

.link-btn:hover {
  background: #111111;
  color: #ffffff;
  text-decoration: none;
}

.link-btn svg {
  width: 16px;
  height: 16px;
  fill: currentColor;
}

.link-btn img {
  width: 18px;
  height: 18px;
}

/* Sections */
.section {
  padding: 56px 0;
}

.section-alt {
  background: var(--alt);
  border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
}

.section-title {
  font-family: 'Google Sans', 'Noto Sans', sans-serif;
  font-size: 28px;
  font-weight: 600;
  text-align: center;
  margin: 0 0 24px;
}

.section-lead {
  text-align: center;
  color: var(--muted);
  margin: -12px auto 28px;
  max-width: 640px;
}

.abstract {
  text-align: justify;
  hyphens: auto;
  margin: 0;
}

/* Figure */
.figure {
  margin: 0;
}

.figure img {
  display: block;
  width: 100%;
  height: auto;
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 6px;
}

.figure figcaption {
  max-width: 860px;
  margin: 18px auto 0;
  font-size: 14.5px;
  line-height: 1.65;
  color: #3c4043;
  text-align: justify;
}

/* Samples toolbar */
.toolbar {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
  gap: 16px 32px;
  padding: 14px 18px;
  margin-bottom: 28px;
  border: 1px solid var(--line);
  border-radius: 8px;
}

.toolbar-group {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.toolbar-label {
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.6px;
  text-transform: uppercase;
  color: var(--muted);
}

.stepper {
  display: flex;
  align-items: center;
  gap: 6px;
}

.step-btn {
  width: 32px;
  height: 32px;
  border: 1px solid #d0d0d0;
  border-radius: 6px;
  background: #ffffff;
  color: var(--ink);
  font-size: 18px;
  line-height: 1;
  cursor: pointer;
}

.step-btn:hover:not(:disabled) {
  border-color: var(--ink);
}

.step-btn:disabled {
  color: #c4c4c4;
  cursor: default;
}

.example-select {
  height: 32px;
  padding: 0 10px;
  border: 1px solid #d0d0d0;
  border-radius: 6px;
  background: #ffffff;
  color: var(--ink);
  font-size: 14px;
  font-variant-numeric: tabular-nums;
}

.chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.chip {
  padding: 4px 12px;
  border: 1px solid #d0d0d0;
  border-radius: 999px;
  background: #ffffff;
  color: var(--muted);
  font-size: 13.5px;
  cursor: pointer;
  transition: all 0.15s ease;
}

.chip:hover {
  border-color: var(--ink);
}

.chip.active {
  background: #363636;
  border-color: #363636;
  color: #ffffff;
}

.chip.ours.active {
  background: var(--accent);
  border-color: var(--accent);
}

.step-btn:focus-visible,
.chip:focus-visible,
.example-select:focus-visible,
.copy-btn:focus-visible,
.link-btn:focus-visible {
  outline: 2px solid var(--link);
  outline-offset: 2px;
}

/* Audio */
.sample-group + .sample-group {
  margin-top: 24px;
}

.group-label {
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.6px;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 10px;
  padding-bottom: 6px;
  border-bottom: 1px solid var(--line);
}

.audio-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
  gap: 14px;
}

.audio-card {
  padding: 12px 14px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #ffffff;
}

.audio-card-ours {
  border-color: var(--accent);
  box-shadow: 0 0 0 1px var(--accent) inset;
}

.audio-name {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14.5px;
  font-weight: 600;
  margin-bottom: 8px;
}

.audio-meta {
  margin: -6px 0 8px;
  font-size: 12.5px;
  color: var(--muted);
  font-variant-numeric: tabular-nums;
}

.ours-tag {
  font-size: 11px;
  font-weight: 600;
  padding: 1px 8px;
  border-radius: 999px;
  background: var(--accent);
  color: #ffffff;
}

.audio-card audio {
  display: block;
  width: 100%;
  height: 36px;
}

/* BibTeX */
.bibtex-header {
  position: relative;
}

.copy-btn {
  position: absolute;
  right: 0;
  top: 50%;
  transform: translateY(-50%);
  padding: 4px 14px;
  border: 1px solid #d0d0d0;
  border-radius: 6px;
  background: #ffffff;
  color: var(--ink);
  font-size: 13px;
  cursor: pointer;
}

.copy-btn:hover {
  border-color: var(--ink);
}

.bibtex-header .section-title {
  margin-bottom: 20px;
}

.bibtex {
  margin: 0;
  padding: 18px 22px;
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 8px;
  font-size: 13.5px;
  line-height: 1.6;
  color: var(--ink);
  overflow-x: auto;
}

/* Footer */
.footer {
  padding: 32px 0 40px;
  text-align: center;
  font-size: 13px;
  color: var(--faint);
}

.footer a {
  color: var(--muted);
}

/* Responsive */
@media (max-width: 640px) {
  .hero {
    padding: 48px 0 32px;
  }

  .title {
    font-size: 28px;
  }

  .authors {
    font-size: 16px;
  }

  .section {
    padding: 40px 0;
  }

  .section-title {
    font-size: 24px;
  }

  .abstract,
  .figure figcaption {
    text-align: left;
  }

  .toolbar {
    flex-direction: column;
    align-items: flex-start;
  }

  .copy-btn {
    top: 0;
    transform: none;
  }
}
</style>
