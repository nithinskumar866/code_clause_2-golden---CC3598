import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, fireEvent, cleanup } from '@testing-library/react';
import { ToastProvider } from '../../components/ui/Toast';
import { History } from './History';
import * as historyApi from '../../api/history';
import type { HistoryPage, HistoryRecord } from '../../types';

vi.mock('../../api/history');

const fetchHistoryMock = vi.mocked(historyApi.fetchHistory);
const deleteHistoryMock = vi.mocked(historyApi.deleteHistoryItem);

const record = (id: number): HistoryRecord => ({
  id,
  created_at: '2026-02-01T10:00:00Z',
  resume_id: id,
  jd_id: id,
  resume_filename: `resume_${id}.pdf`,
  jd_filename: `jd_${id}.docx`,
  overall_score: 80,
  recruiter_recommendation: 'Proceed to interview',
  summary: 'Strong match on core requirements.',
});

const pageOf = (items: HistoryRecord[], over: Partial<HistoryPage['meta']> = {}): HistoryPage => ({
  items,
  meta: { total_count: items.length, page: 1, page_size: 8, total_pages: 1, ...over },
});

const renderHistory = () =>
  render(
    <ToastProvider>
      <History onOpenCandidate={vi.fn()} />
    </ToastProvider>,
  );

beforeEach(() => {
  vi.clearAllMocks();
  window.history.replaceState(null, '', '/');
});

afterEach(() => cleanup());

describe('History page', () => {
  it('loads and renders analyses from the API with the default query', async () => {
    fetchHistoryMock.mockResolvedValue(pageOf([record(1), record(2)]));
    renderHistory();

    expect(await screen.findByText('resume_1.pdf')).toBeInTheDocument();
    expect(screen.getByText('resume_2.pdf')).toBeInTheDocument();
    expect(fetchHistoryMock).toHaveBeenCalledWith(
      expect.objectContaining({ sort: 'newest', page: 1, page_size: 8 }),
    );
  });

  it('shows the empty state when there are no analyses', async () => {
    fetchHistoryMock.mockResolvedValue(pageOf([], { total_count: 0, total_pages: 0 }));
    renderHistory();

    expect(await screen.findByText('No analyses yet')).toBeInTheDocument();
  });

  it('maps a recommendation filter to the API query and the URL, resetting page', async () => {
    fetchHistoryMock.mockResolvedValue(pageOf([record(1)]));
    renderHistory();
    await screen.findByText('resume_1.pdf');

    fireEvent.change(screen.getByLabelText('Recommendation'), { target: { value: 'Selected' } });

    await waitFor(() =>
      expect(fetchHistoryMock).toHaveBeenLastCalledWith(
        expect.objectContaining({ recommendation: 'Selected', page: 1 }),
      ),
    );
    expect(new URLSearchParams(window.location.search).get('recommendation')).toBe('Selected');
  });

  it('deletes an analysis after confirmation (existing behavior)', async () => {
    fetchHistoryMock.mockResolvedValue(pageOf([record(1)]));
    deleteHistoryMock.mockResolvedValue();
    renderHistory();
    await screen.findByText('resume_1.pdf');

    fireEvent.click(screen.getByRole('button', { name: /delete analysis for resume_1\.pdf/i }));
    fireEvent.click(screen.getByRole('button', { name: 'Delete' }));

    await waitFor(() => expect(deleteHistoryMock).toHaveBeenCalledWith(1));
  });
});
