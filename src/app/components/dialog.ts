/**
 * Modal dialogs (alert, confirm, prompt) — accessible (role=alertdialog, aria-modal, focus
 * management, Escape) replacements for window.confirm/prompt, which look foreign in a Home
 * Screen app.
 */
import { h } from '../dom';
import { overlayLayer, pushModal, registerCloser } from './layers';

export interface ConfirmOptions {
  title: string;
  message?: string;
  confirmLabel?: string;
  cancelLabel?: string;
  destructive?: boolean;
}

let seq = 0;

function openDialog<T>(build: (done: (v: T) => void) => { body: HTMLElement[]; actions: HTMLElement[] }, title: string, cancelValue: T, role = 'alertdialog'): Promise<T> {
  return new Promise<T>((resolve) => {
    const id = `dlg-${++seq}`;
    let finished = false;
    const backdrop = h('div', { class: 'dialog-backdrop' });
    const done = (v: T): void => {
      if (finished) return;
      finished = true;
      release();
      unregister();
      backdrop.classList.remove('show');
      dialog.classList.remove('show');
      setTimeout(() => wrap.remove(), 200);
      resolve(v);
    };
    const { body, actions } = build(done);
    const dialog = h(
      'div',
      { class: 'dialog', role, 'aria-modal': 'true', 'aria-labelledby': `${id}-t`, tabindex: '-1' },
      h('h2', { class: 'dialog-title', id: `${id}-t`, text: title }),
      ...body,
      h('div', { class: 'dialog-actions' }, ...actions),
    );
    const wrap = h('div', { class: 'dialog-wrap' }, backdrop, dialog);
    backdrop.addEventListener('click', () => done(cancelValue));
    overlayLayer().appendChild(wrap);
    const release = pushModal(dialog, () => done(cancelValue));
    const unregister = registerCloser(() => done(cancelValue));
    requestAnimationFrame(() => {
      backdrop.classList.add('show');
      dialog.classList.add('show');
    });
  });
}

export function confirmDialog(o: ConfirmOptions): Promise<boolean> {
  return openDialog<boolean>(
    (done) => {
      const cancel = h('button', { type: 'button', class: 'btn btn-secondary btn-md', text: o.cancelLabel ?? 'Cancel' });
      const ok = h('button', {
        type: 'button',
        class: `btn ${o.destructive ? 'btn-danger' : 'btn-primary'} btn-md`,
        text: o.confirmLabel ?? 'OK',
        'data-testid': 'dialog-confirm',
      });
      cancel.addEventListener('click', () => done(false));
      ok.addEventListener('click', () => done(true));
      if (!o.destructive) ok.setAttribute('autofocus', '');
      else cancel.setAttribute('autofocus', '');
      return { body: o.message ? [h('p', { class: 'dialog-message', text: o.message })] : [], actions: [cancel, ok] };
    },
    o.title,
    false,
  );
}

export function alertDialog(title: string, message?: string, okLabel = 'OK'): Promise<void> {
  return openDialog<void>(
    (done) => {
      const ok = h('button', { type: 'button', class: 'btn btn-primary btn-md', text: okLabel, autofocus: '' });
      ok.addEventListener('click', () => done());
      return { body: message ? [h('p', { class: 'dialog-message', text: message })] : [], actions: [ok] };
    },
    title,
    undefined,
  );
}

export function promptDialog(o: { title: string; message?: string; value?: string; placeholder?: string; confirmLabel?: string; maxLength?: number }): Promise<string | null> {
  return openDialog<string | null>(
    (done) => {
      // font-size ≥ 16 px in CSS: iOS does not zoom into the field.
      const input = h('input', {
        type: 'text',
        class: 'text-input',
        placeholder: o.placeholder ?? '',
        maxlength: String(o.maxLength ?? 80),
        autocomplete: 'off',
        enterkeyhint: 'done',
        autofocus: '',
        'aria-label': o.title,
      });
      input.value = o.value ?? '';
      const submit = (): void => {
        const v = input.value.trim();
        done(v.length ? v : null);
      };
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          submit();
        }
      });
      const cancel = h('button', { type: 'button', class: 'btn btn-secondary btn-md', text: 'Cancel' });
      const ok = h('button', { type: 'button', class: 'btn btn-primary btn-md', text: o.confirmLabel ?? 'Save', 'data-testid': 'dialog-confirm' });
      cancel.addEventListener('click', () => done(null));
      ok.addEventListener('click', submit);
      requestAnimationFrame(() => {
        input.focus();
        input.select();
      });
      const body: HTMLElement[] = [];
      if (o.message) body.push(h('p', { class: 'dialog-message', text: o.message }));
      body.push(input);
      return { body, actions: [cancel, ok] };
    },
    o.title,
    null,
    'dialog',
  );
}
