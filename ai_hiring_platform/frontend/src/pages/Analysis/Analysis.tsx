import { useState, useEffect, type FC } from 'react';
import { Database } from 'lucide-react';
import axios from 'axios';
import type { AnalysisReport, FileRecord } from '../../types';
import { ConfigPanel } from '../../components/analysis/ConfigPanel';
import { ReportHeader } from '../../components/analysis/ReportHeader';
import type { ReportViewMode } from '../../components/analysis/ReportHeader';
import { HiringReportView } from '../../components/analysis/HiringReportView';
import { RawRagView } from '../../components/analysis/RawRagView';

export const Analysis: FC = () => {
  const [resumes, setResumes] = useState<FileRecord[]>([]);
  const [jds, setJds] = useState<FileRecord[]>([]);
  const [selectedResume, setSelectedResume] = useState<string>('');
  const [selectedJd, setSelectedJd] = useState<string>('');

  const [loadingLists, setLoadingLists] = useState(false);
  const [evaluating, setEvaluating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<AnalysisReport | null>(null);
  const [viewMode, setViewMode] = useState<ReportViewMode>('report');

  const fetchData = async () => {
    setLoadingLists(true);
    setError(null);
    try {
      const [resumesRes, jdsRes] = await Promise.all([
        axios.get('http://localhost:8000/api/v1/resume'),
        axios.get('http://localhost:8000/api/v1/job'),
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
      const response = await axios.post(
        `http://localhost:8000/api/v1/analysis/evaluate?resume_id=${selectedResume}&jd_id=${selectedJd}`,
      );
      if (response.data && response.data.success) {
        setReport(response.data.data.report);
      } else {
        setError(response.data?.message || 'Analysis evaluation failed.');
      }
    } catch (err: any) {
      console.error(err);
      const detailMsg =
        err.response?.data?.detail ||
        err.response?.data?.error?.details ||
        err.message ||
        'Failed to trigger evaluation engine';
      setError(detailMsg);
    } finally {
      setEvaluating(false);
    }
  };

  const handleDownloadJSON = () => {
    if (!report) return;
    const jsonString = `data:text/json;charset=utf-8,${encodeURIComponent(JSON.stringify(report, null, 2))}`;
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', jsonString);
    downloadAnchor.setAttribute('download', `explainable_hiring_report_${report.analysis_id}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-white">Explainable Hiring Decision Engine</h1>
        <p className="mt-2 text-sm text-gray-400">
          Orchestrate local RAG retrieval and explainable evaluation reasoning dynamically via LangGraph state graphs.
        </p>
      </div>

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
            <div className="space-y-6 animate-fadeIn">
              <ReportHeader
                report={report}
                viewMode={viewMode}
                onChangeView={setViewMode}
                onDownload={handleDownloadJSON}
              />

              {viewMode === 'report' ? (
                <HiringReportView report={report} />
              ) : (
                <RawRagView results={report.retrieval_results} />
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
