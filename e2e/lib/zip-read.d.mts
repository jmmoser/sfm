export function crc32(bytes: Uint8Array): number;
export function safeEntryName(name: string): string;
export function readStoreZip(buf: Uint8Array | ArrayBuffer): { name: string; data: Uint8Array }[];
