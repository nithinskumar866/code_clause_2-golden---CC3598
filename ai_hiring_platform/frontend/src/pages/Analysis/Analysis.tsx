import { useState, useEffect, type FC } from 'react';
import { Database } from 'lucide-react';
import { api } from '../../api/client';
import type { AnalysisReport, FileRecord } from '../../types';
import { ConfigPanel } from '../../components/analysis/ConfigPanel';
import { ReportViewer } from '../../components/analysis/ReportViewer';
import { PageHeader } from '../../components/ui/PageHeader';
import { useToast } from '../../components/ui/toast-context';

export const Analysis: FC = () => {
  const [resumes, setResumes] = useState<FileRecord[]>([]);
  const [jds, setJds] = useState<FileRecord[]>([]);
  const [selectedResume, setSelectedResume] = useState<string>('');
  const [selectedJd, setSelectedJd] = useState<string>('');

  const [loadingLists, setLoadingLists] = useState(false);
  const [evaluating, setEvaluating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<AnalysisReport | null>(null);

  const toast = useToast();

  const fetchData = async () => {
    setLoadingLists(true);
    setError(null);
    try {
      const [resumesRes, jdsRes] = await Promise.all([
        api.get('/resume'),
        api.get('/job'),
      ]);

      if (resumesRes.data && resumesRes.data.success) {
        setResumes(resumesRes.data.data);
      }
      if (jdsRes.data && jdsRes.data.success) {
        setJds(jdsRes.data.data);
      }
    } catch (err: any) {
      console.error(err);
      setError('Failed to fetch database files. Ensure the FastAPI backend is running.');
    } finally {
      setLoadingLists(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleEvaluate = async () => {
    if (!selectedResume || !selectedJd) return;
    setEvaluating(true);
    setError(null);
    setReport(null);

    try {
      const response = await api.post(
        `/analysis/evaluate?resume_id=${selectedResume}&jd_id=${selectedJd}`,
      );
      if (response.data && response.data.success) {
        const rpt = response.data.data.report;
        setReport(rpt);
        toast.success('Evaluation complete', `Overall match score: ${rpt.overall_score}%`);
      } else {
        const msg = response.data?.message || 'Analysis evaluation failed.';
        setError(msg);
        toast.error(msg);
      }
    } catch (err: any) {
      console.error(err);
      const detailMsg =
        err.response?.data?.detail ||
        err.response?.data?.error?.details ||
        err.message ||
        'Failed to trigger evaluation engine';
      setError(detailMsg);
      toast.error(detailMsg);
    } finally {
      setEvaluating(false);
    }
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      <PageHeader
        title="AI Analysis"
        description="Select a candidate and a job description, then run the explainable evaluation pipeline."
      />

      <div className="grid grid-cols-1 gap-8 lg:grid-cols-3">
        <ConfigPanel
          resumes={resumes}
          jds={jds}
          selectedResume={selectedResume}
          selectedJd={selectedJd}
          onSelectResume={setSelectedResume}
          onSelectJd={setSelectedJd}
          loadingLists={loadingLists}
          evaluating={evaluating}
          error={error}
          onEvaluate={handleEvaluate}
          onRefresh={fetchData}
        />

        <div className="lg:col-span-2 space-y-6">
          {!report ? (
            <div className="flex flex-col items-center justify-center border border-white/5 rounded-xl bg-card py-44 px-6 text-center h-full">
              <Database className="h-10 w-10 text-gray-500 mb-4" />
              <h3 className="text-base font-medium text-white">No Evaluation Triggered</h3>
              <p className="text-gray-400 text-xs mt-1 max-w-sm">
                Select a candidate resume and job description, then run the evaluation pipeline to invoke local FAISS
                indexing and LLM reasoning.
              </p>
            </div>
          ) : (
            <div className="animate-fadeIn">
              <ReportViewer report={report} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
