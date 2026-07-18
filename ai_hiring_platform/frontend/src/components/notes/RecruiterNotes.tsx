import { useCallback, useEffect, useState, type FC } from 'react';
import { StickyNote, Plus, Pencil, Trash2, X, Check } from 'lucide-react';
import type { Note } from '../../types';
import { listNotes, createNote, updateNote, deleteNote } from '../../api/notes';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';
import { Skeleton } from '../common/Skeleton';
import { ErrorState } from '../ui/ErrorState';
import { useToast } from '../ui/toast-context';

interface RecruiterNotesProps {
  analysisId: number;
}

const formatWhen = (iso: string): string => {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleString();
};

/** Full recruiter-notes CRUD for one analysis: add, list, edit, delete (persisted). */
export const RecruiterNotes: FC<RecruiterNotesProps> = ({ analysisId }) => {
  const [notes, setNotes] = useState<Note[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [author, setAuthor] = useState('');
  const [text, setText] = useState('');
  const [adding, setAdding] = useState(false);

  const [editingId, setEditingId] = useState<number | null>(null);
  const [editText, setEditText] = useState('');
  const [busyId, setBusyId] = useState<number | null>(null);

  const toast = useToast();

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setNotes(await listNotes(analysisId));
    } catch {
      setError('Failed to load recruiter notes.');
    } finally {
      setLoading(false);
    }
  }, [analysisId]);

  useEffect(() => {
    load();
  }, [load]);

  const add = async () => {
    if (!text.trim() || !author.trim()) {
      toast.error('Both author and note text are required.');
      return;
    }
    setAdding(true);
    try {
      const created = await createNote(analysisId, text.trim(), author.trim());
      setNotes((prev) => [created, ...prev]);
      setText('');
      toast.success('Note added');
    } catch (err: any) {
      toast.error(err?.message || 'Failed to add note');
    } finally {
      setAdding(false);
    }
  };

  const startEdit = (note: Note) => {
    setEditingId(note.id);
    setEditText(note.text);
  };

  const saveEdit = async (note: Note) => {
    if (!editText.trim()) {
      toast.error('Note text cannot be empty.');
      return;
    }
    setBusyId(note.id);
    try {
      const updated = await updateNote(note.id, { text: editText.trim() });
      setNotes((prev) => prev.map((n) => (n.id === note.id ? updated : n)));
      setEditingId(null);
      toast.success('Note updated');
    } catch (err: any) {
      toast.error(err?.message || 'Failed to update note');
    } finally {
      setBusyId(null);
    }
  };

  const remove = async (note: Note) => {
    setBusyId(note.id);
    try {
      await deleteNote(note.id);
      setNotes((prev) => prev.filter((n) => n.id !== note.id));
      toast.success('Note deleted');
    } catch (err: any) {
      toast.error(err?.message || 'Failed to delete note');
    } finally {
      setBusyId(null);
    }
  };

  return (
    <Card className="p-6 space-y-4">
      <h3 className="flex items-center gap-2 border-b border-white/5 pb-2 text-sm font-semibold text-white">
        <StickyNote className="h-4 w-4 text-indigo-400" /> Recruiter Notes
        {!loading && <span className="text-xs font-normal text-gray-500">({notes.length})</span>}
      </h3>

      {/* Add form */}
      <div className="space-y-2">
        <input
          value={author}
          onChange={(e) => setAuthor(e.target.value)}
          placeholder="Your name"
          maxLength={255}
          className="w-full rounded-lg border border-white/10 bg-black/40 px-3 py-2 text-sm text-white placeholder-gray-600 focus:border-indigo-500 focus:outline-none"
        />
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Add a note about this candidate…"
          rows={2}
          maxLength={5000}
          className="w-full resize-y rounded-lg border border-white/10 bg-black/40 px-3 py-2 text-sm text-white placeholder-gray-600 focus:border-indigo-500 focus:outline-none"
        />
        <div className="flex justify-end">
          <Button size="sm" leftIcon={<Plus className="h-4 w-4" />} loading={adding} onClick={add}>
            Add note
          </Button>
        </div>
      </div>

      {/* List */}
      {loading ? (
        <Skeleton className="h-20 w-full" />
      ) : error ? (
        <ErrorState message={error} onRetry={load} />
      ) : notes.length === 0 ? (
        <p className="py-2 text-center text-xs italic text-gray-500">No notes yet. Add the first one above.</p>
      ) : (
        <ul className="space-y-3">
          {notes.map((note) => (
            <li key={note.id} className="rounded-lg border border-white/5 bg-black/20 p-3">
              <div className="mb-1 flex items-center justify-between">
                <span className="text-xs font-semibold text-indigo-300">{note.author}</span>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-gray-500">{formatWhen(note.updated_at)}</span>
                  {editingId === note.id ? (
                    <>
                      <button
                        onClick={() => saveEdit(note)}
                        disabled={busyId === note.id}
                        className="text-emerald-400 hover:text-emerald-300 disabled:opacity-50"
                        title="Save"
                      >
                        <Check className="h-3.5 w-3.5" />
                      </button>
                      <button
                        onClick={() => setEditingId(null)}
                        className="text-gray-400 hover:text-white"
                        title="Cancel"
                      >
                        <X className="h-3.5 w-3.5" />
                      </button>
                    </>
                  ) : (
                    <>
                      <button
                        onClick={() => startEdit(note)}
                        className="text-gray-400 hover:text-white"
                        title="Edit"
                      >
                        <Pencil className="h-3.5 w-3.5" />
                      </button>
                      <button
                        onClick={() => remove(note)}
                        disabled={busyId === note.id}
                        className="text-rose-400 hover:text-rose-300 disabled:opacity-50"
                        title="Delete"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </>
                  )}
                </div>
              </div>
              {editingId === note.id ? (
                <textarea
                  value={editText}
                  onChange={(e) => setEditText(e.target.value)}
                  rows={2}
                  maxLength={5000}
                  className="w-full resize-y rounded-lg border border-white/10 bg-black/40 px-3 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none"
                />
              ) : (
                <p className="whitespace-pre-wrap text-sm text-gray-300">{note.text}</p>
              )}
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
};
