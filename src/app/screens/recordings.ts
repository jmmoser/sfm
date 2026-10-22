/**
 * Diagnostics › "Raw capture recordings" (the recordings of Setup › Options › Record raw capture,
 * src/app/recorder.ts): every stored recording with its frames, size, date and state (complete,
 * or "interrupted" when the app was closed while it recorded — still exportable up to the last
 * flush), Export and Delete.
 *
 * Export sheet: the recording as ZIP parts of ≤ 200 MB (src/io/recording-store.ts
 * prepareRecordingExport: the iOS share sheet reads every file fully into memory) — each part
 * "Prepare" (builds the archive; it references the stored JPEGs) then "Share / Save" (saveFile
 * directly in that tap: navigator.share needs a fresh activation). Location is left out unless
 * "Include location" is on (privacy: GPS of the user's home). On a computer: extract every part
 * into one folder, or `node e2e/import-recording.mjs <parts…>` (e2e/README.md).
 */
import { saveFile } from '../../io/download';
import { RecordingStore, prepareRecordingExport, type RecordingExport, type RecordingExportPart, type StoredRecording } from '../../io/recording-store';
import type { App } from '../app';
import { button, chip, iconButton, section, spinner, switchRow } from '../components/controls';
import { confirmDialog } from '../components/dialog';
import { openSheet } from '../components/sheet';
import { toast } from '../components/toast';
import { h, replaceChildren } from '../dom';
import { formatBytes, formatDuration, formatRelativeDate, qualityLabel } from '../format';
import { icon } from '../icons';

type PartState = { status: 'idle' } | { status: 'building' } | { status: 'ready'; blob: Blob } | { status: 'error'; error: string };

export interface RecordingsPanel {
  readonly el: HTMLElement;
  refresh(): Promise<void>;
  dispose(): void;
}

export function createRecordingsPanel(app: App): RecordingsPanel {
  const el = h('div', { class: 'recordings', 'data-testid': 'recordings' });
  let closed = false;
  let store: RecordingStore | null = null;
  let seq = 0;

  const intro = (): HTMLElement => h('p', { class: 'muted', text: 'Frames and sensor readings recorded for replay on a computer (Setup › Options › Record raw capture).' });

  async function openStore(): Promise<RecordingStore> {
    store ??= await RecordingStore.open();
    return store;
  }

  async function refresh(): Promise<void> {
    const my = ++seq;
    replaceChildren(el, section('Raw capture recordings', intro(), h('div', { class: 'list-loading' }, spinner('sm', 'Loading recordings'))));
    let list: StoredRecording[];
    try {
      list = await (await openStore()).list();
    } catch (e) {
      if (!closed && my === seq) replaceChildren(el, section('Raw capture recordings', h('p', { class: 'muted', text: `Recordings unavailable: ${(e as Error)?.message ?? e}` })));
      return;
    }
    if (closed || my !== seq) return;
    const rows = list.map((r) => row(r));
    replaceChildren(
      el,
      section(
        'Raw capture recordings',
        intro(),
        rows.length ? h('div', { class: 'export-list' }, ...rows) : h('p', { class: 'muted', 'data-testid': 'recordings-empty', text: 'None yet. Turn on “Record raw capture” under Options on the New scan screen.' }),
      ),
    );
  }

  function row(r: StoredRecording): HTMLElement {
    const hd = r.header;
    const frames = hd.summary?.frames;
    const interrupted = hd.state !== 'complete';
    const sub = [
      `${formatRelativeDate(hd.createdAt)} · ${qualityLabel(hd.engine.quality)}`,
      frames !== undefined ? `${frames} frames` : null,
      hd.summary ? formatDuration(hd.summary.durationMs) : null,
      formatBytes(r.bytes),
    ]
      .filter(Boolean)
      .join(' · ');
    return h(
      'div',
      { class: 'export-item', 'data-testid': 'recording-item' },
      h('span', { class: 'export-icon' }, icon('storage', { size: 22 })),
      h(
        'div',
        { class: 'export-text' },
        h('span', { class: 'export-title', text: hd.name }),
        h('span', { class: 'export-sub', text: sub }),
        interrupted ? chip('Interrupted', 'warn') : null,
      ),
      button({ label: 'Export', variant: 'secondary', size: 'sm', testId: 'recording-export', onClick: () => openExport(r) }),
      iconButton('trash', 'Delete recording', () => void remove(r), { testId: 'recording-delete' }),
    );
  }

  async function remove(r: StoredRecording): Promise<void> {
    const ok = await confirmDialog({ title: 'Delete recording?', message: `“${r.header.name}” (${formatBytes(r.bytes)}) is removed from this device.`, confirmLabel: 'Delete', destructive: true });
    if (!ok || closed) return;
    try {
      await (await openStore()).delete(r);
      toast(`Deleted (${formatBytes(r.bytes)} freed)`, { tone: 'good' });
    } catch (e) {
      toast(`Could not delete the recording: ${(e as Error)?.message ?? e}`, { tone: 'bad' });
    }
    if (!closed) await refresh();
  }

  function openExport(r: StoredRecording): void {
    let includeLocation = false;
    let exp: RecordingExport | null = null;
    let error: string | null = null;
    const parts = new Map<number, PartState>();
    let sharingUntil = 0;
    let gen = 0;
    const list = h('div', { class: 'export-list' });
    const locRow = switchRow({
      label: 'Include location',
      description: 'GPS fixes of the capture (needed to replay GPS scale and north; reveals where it was recorded)',
      icon: 'location',
      checked: false,
      testId: 'recording-include-location',
      onChange: (v) => {
        includeLocation = v;
        void plan();
      },
    });
    const note = h('p', { class: 'muted', text: 'On a computer: extract every part into one folder (or node e2e/import-recording.mjs <parts…>), then node e2e/run-e2e.mjs <folder name>.' });
    openSheet({ title: 'Export recording', subtitle: r.header.name, content: [h('div', { class: 'group' }, locRow.el), list, note], testId: 'sheet-recording-export' });

    async function plan(): Promise<void> {
      const my = ++gen;
      exp = null;
      error = null;
      parts.clear();
      render();
      try {
        const e = await prepareRecordingExport(await openStore(), r, { stripLocation: !includeLocation });
        if (my !== gen) return;
        exp = e;
      } catch (e) {
        if (my !== gen) return;
        error = (e as Error)?.message ?? String(e);
      }
      render();
    }

    function render(): void {
      if (error) {
        replaceChildren(list, h('p', { class: 'muted', text: `Cannot export: ${error}` }));
        return;
      }
      if (!exp) {
        replaceChildren(list, h('div', { class: 'list-loading' }, spinner('sm', 'Reading the recording')));
        return;
      }
      if (exp.meta.numFrames === 0) {
        replaceChildren(list, h('p', { class: 'muted', text: 'This recording has no frames.' }));
        return;
      }
      replaceChildren(list, ...exp.parts.map((p) => partRow(p)));
    }

    function partRow(p: RecordingExportPart): HTMLElement {
      const st = parts.get(p.part) ?? { status: 'idle' };
      let action: HTMLElement;
      if (st.status === 'building') action = h('span', { class: 'export-building' }, spinner('sm'), h('span', { text: 'Preparing…' }));
      else if (st.status === 'ready') {
        const blob = st.blob;
        action = button({
          label: app.env.share ? 'Share / Save' : 'Save',
          icon: app.env.share ? 'share' : 'download',
          variant: 'primary',
          size: 'sm',
          testId: `recording-save-${p.part}`,
          // fresh user activation: saveFile is called synchronously in this tap
          onClick: () => {
            if (performance.now() < sharingUntil) return;
            sharingUntil = performance.now() + 2000;
            saveFile(blob, p.fileName)
              .then(
                (res) => {
                  if (res === 'shared') toast('Shared', { tone: 'good' });
                  else if (res === 'downloaded') toast(`Saved ${p.fileName}`, { tone: 'good', icon: 'download' });
                },
                (err) => {
                  const n = (err as { name?: string })?.name;
                  if (n === 'NotAllowedError') toast('Tap “Share / Save” again to save the file.', { tone: 'warn' });
                  else if (n !== 'InvalidStateError') toast(`Could not save: ${(err as Error)?.message ?? err}`, { tone: 'bad' });
                },
              )
              .finally(() => {
                sharingUntil = 0;
              });
          },
        });
      } else action = button({ label: st.status === 'error' ? 'Retry' : 'Prepare', variant: 'secondary', size: 'sm', testId: `recording-prepare-${p.part}`, onClick: () => void prepare(p) });
      const sub = st.status === 'error' ? st.error : `${p.fileName} · ${formatBytes(p.bytes)} · ${p.entries} files${p.part === 1 ? ' (with the metadata)' : ''}`;
      return h(
        'div',
        { class: `export-item${st.status === 'error' ? ' error' : ''}`, 'data-testid': `recording-part-${p.part}` },
        h('span', { class: 'export-icon' }, icon('zip', { size: 22 })),
        h('div', { class: 'export-text' }, h('span', { class: 'export-title', text: p.count > 1 ? `Part ${p.part} of ${p.count}` : 'Recording ZIP' }), h('span', { class: 'export-sub', text: sub })),
        action,
      );
    }

    async function prepare(p: RecordingExportPart): Promise<void> {
      const my = gen;
      // one prepared part at a time: each holds up to 200 MB once shared
      for (const [k, v] of parts) if (v.status === 'ready') parts.delete(k);
      parts.set(p.part, { status: 'building' });
      render();
      try {
        const blob = await p.build();
        if (my === gen) parts.set(p.part, { status: 'ready', blob });
      } catch (e) {
        if (my === gen) parts.set(p.part, { status: 'error', error: (e as Error)?.message ?? String(e) });
      }
      if (my === gen && !closed) render();
    }

    void plan();
  }

  return {
    el,
    refresh,
    dispose() {
      closed = true;
      store?.close();
      store = null;
    },
  };
}
