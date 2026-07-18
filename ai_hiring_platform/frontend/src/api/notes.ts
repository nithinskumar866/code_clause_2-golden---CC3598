import { api } from './client';
import type { Note } from '../types';

/** List recruiter notes for an analysis, newest first. */
export async function listNotes(analysisId: number): Promise<Note[]> {
  const res = await api.get(`/analysis/${analysisId}/notes`);
  if (res.data?.success) return res.data.data as Note[];
  throw new Error(res.data?.message || 'Failed to load notes');
}

/** Create a recruiter note on an analysis. */
export async function createNote(analysisId: number, text: string, author: string): Promise<Note> {
  const res = await api.post(`/analysis/${analysisId}/notes`, { text, author });
  if (res.data?.success) return res.data.data as Note;
  throw new Error(res.data?.message || 'Failed to create note');
}

/** Update a note's text and/or author. */
export async function updateNote(
  noteId: number,
  patch: { text?: string; author?: string },
): Promise<Note> {
  const res = await api.put(`/notes/${noteId}`, patch);
  if (res.data?.success) return res.data.data as Note;
  throw new Error(res.data?.message || 'Failed to update note');
}

/** Delete a note. */
export async function deleteNote(noteId: number): Promise<void> {
  const res = await api.delete(`/notes/${noteId}`);
  if (!res.data?.success) throw new Error(res.data?.message || 'Failed to delete note');
}
