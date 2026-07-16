import React, { useState, useEffect } from 'react';
import { Play, FileText, CheckCircle2, AlertCircle, RefreshCw, Layers, Database, HelpCircle, Mail, BookOpen, Copy, Check, Download, Brain, Award } from 'lucide-react';
import axios from 'axios';

interface FileRecord {
  id: number;
  filename: string;
  upload_time: string;
  status: string;
}

interface RequirementFit {
  requirement: string;
  category: string;
  status: string; // "Matched" | "Partial" | "Missing"
  matched_evidence: string;
  explanation: string;
  limitations: string;
  confidence: number;
}

interface LearningRoadmapItem {
  skill: string;
  estimated_time: string;
  reason: string;
}

interface MatchRecord {
  chunk: string;
  section: string;
  score: number;
  confidence: number;
  page: number;
  filename: string;
  chunk_id: number;
}

interface RetrievalResult {
  requirement: string;
  matches: MatchRecord[];
  error?: string;
}

interface AnalysisReport {
  analysis_id: number;
  candidate_id: number;
  resume_id: number;
  jd_id: number;
  retrieval_results: RetrievalResult[];
  overall_score: number;
  coverage_score: number;
  experience_score: number;
  project_score: number;
  confidence_score: number;
  quality_score: number;
  
  summary: string;
  requirements: RequirementFit[];
  
  strengths: string[];
  weaknesses: string[];
  skill_relationships: string[];
  
  missing_skills: string[];
  learning_roadmap: LearningRoadmapItem[];
  interview_questions: string[];
  
  recruiter_recommendation: string;
  rejection_email: string | null;
}

export const Analysis: React.FC = () => {
  const [resumes, setResumes] = useState<FileRecord[]>([]);
  const [jds, setJds] = useState<FileRecord[]>([]);
  const [selectedResume, setSelectedResume] = useState<string>('');
  const [selectedJd, setSelectedJd] = useState<string>('');
  
  const [loadingLists, setLoadingLists] = useState(false);
  const [evaluating, setEvaluating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<AnalysisReport | null>(null);
  const [viewMode, setViewMode] = useState<'report' | 'raw_rag'>('report');
  
  const [copiedEmail, setCopiedEmail] = useState(false);

  const fetchData = async () => {
    setLoadingLists(true);
    setError(null);
    try {
      const [resumesRes, jdsRes] = await Promise.all([
        axios.get('http://localhost:8000/api/v1/resume'),
        axios.get('http://localhost:8000/api/v1/job')
      ]);
      
      if (resumesRes.data && resumesRes.data.success) {
        setResumes(resumesRes.data.data);
      }
      if (jdsRes.data && jdsRes.data.success) {
        setJds(jdsRes.data.data);
      }
    } catch (err: any) {
      console.error(err);
      setError("Failed to fetch database files. Ensure the FastAPI backend is running.");
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
        `http://localhost:8000/api/v1/analysis/evaluate?resume_id=${selectedResume}&jd_id=${selectedJd}`
      );
      if (response.data && response.data.success) {
        setReport(response.data.data.report);
      } else {
        setError(response.data?.message || "Analysis evaluation failed.");
      }
    } catch (err: any) {
      console.error(err);
      const detailMsg = err.response?.data?.detail || err.response?.data?.error?.details || err.message || "Failed to trigger evaluation engine";
      setError(detailMsg);
    } finally {
      setEvaluating(false);
    }
  };

  const handleCopyEmail = () => {
    if (report?.rejection_email) {
      navigator.clipboard.writeText(report.rejection_email);
      setCopiedEmail(true);
      setTimeout(() => setCopiedEmail(false), 2000);
    }
  };

  const handleDownloadJSON = () => {
    if (!report) return;
    const jsonString = `data:text/json;charset=utf-8,${encodeURIComponent(
      JSON.stringify(report, null, 2)
    )}`;
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", jsonString);
    downloadAnchor.setAttribute("download", `explainable_hiring_report_${report.analysis_id}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const getScoreColor = (score: number) => {
    if (score >= 75) return 'text-emerald-400 border-emerald-500/20 bg-emerald-500/10';
    if (score >= 60) return 'text-amber-400 border-amber-500/20 bg-amber-500/10';
    return 'text-rose-400 border-rose-500/20 bg-rose-500/10';
  };

  const getScoreBarBg = (score: number) => {
    if (score >= 75) return 'bg-emerald-500';
    if (score >= 60) return 'bg-amber-500';
    return 'bg-rose-500';
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'Matched':
        return (
          <span className="inline-flex items-center gap-1 rounded-md bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-0.5 text-xs font-semibold text-emerald-400">
            <CheckCircle2 className="h-3 w-3" /> Matched
          </span>
        );
      case 'Partial':
        return (
          <span className="inline-flex items-center gap-1 rounded-md bg-amber-500/10 border border-amber-500/20 px-2.5 py-0.5 text-xs font-semibold text-amber-400">
            <HelpCircle className="h-3 w-3" /> Partial
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1 rounded-md bg-rose-500/10 border border-rose-500/20 px-2.5 py-0.5 text-xs font-semibold text-rose-400">
            <AlertCircle className="h-3 w-3" /> Missing
          </span>
        );
    }
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
        {/* Selection Configuration Panel */}
        <div className="lg:col-span-1 rounded-xl border border-white/5 bg-card p-6 h-fit space-y-6">
          <div className="flex items-center gap-2 pb-2 border-b border-white/5">
            <Layers className="h-5 w-5 text-indigo-400" />
            <h2 className="text-base font-semibold text-white">Configure Analysis</h2>
          </div>

          {error && (
            <div className="flex items-start gap-2.5 rounded-lg border border-rose-500/20 bg-rose-500/10 p-3 text-xs text-rose-400">
              <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}

          <div className="space-y-4">
            {/* Candidate Resume Selector */}
            <div>
              <label className="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                Candidate Resume
              </label>
              <select
                value={selectedResume}
                onChange={(e) => setSelectedResume(e.target.value)}
                disabled={loadingLists || evaluating}
                className="w-full rounded-lg border border-white/10 bg-black/40 px-3.5 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 disabled:opacity-50"
              >
                <option value="">-- Choose Candidate --</option>
                {resumes.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.filename} (ID: #{r.id})
                  </option>
                ))}
              </select>
            </div>

            {/* Job Description Selector */}
            <div>
              <label className="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                Job Description
              </label>
              <select
                value={selectedJd}
                onChange={(e) => setSelectedJd(e.target.value)}
                disabled={loadingLists || evaluating}
                className="w-full rounded-lg border border-white/10 bg-black/40 px-3.5 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 disabled:opacity-50"
              >
                <option value="">-- Choose Job Description --</option>
                {jds.map((j) => (
                  <option key={j.id} value={j.id}>
                    {j.filename} (ID: #{j.id})
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="pt-4 border-t border-white/5 flex gap-2">
            <button
              onClick={handleEvaluate}
              disabled={evaluating || !selectedResume || !selectedJd}
              className="w-full flex items-center justify-center gap-2 rounded-lg bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed transition shadow-lg shadow-indigo-600/10"
            >
              {evaluating ? (
                <>
                  <RefreshCw className="h-4 w-4 animate-spin" />
                  Running LangGraph Workflow...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 fill-current" />
                  Evaluate Candidate Fit
                </>
              )}
            </button>

            <button
              onClick={fetchData}
              disabled={loadingLists || evaluating}
              className="p-2.5 rounded-lg border border-white/10 text-gray-400 hover:text-white hover:bg-white/5 transition"
              title="Refresh Files List"
            >
              <RefreshCw className={`h-4 w-4 ${loadingLists ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>

        {/* Dashboard Panels */}
        <div className="lg:col-span-2 space-y-6">
          {!report ? (
            <div className="flex flex-col items-center justify-center border border-white/5 rounded-xl bg-card py-44 px-6 text-center h-full">
              <Database className="h-10 w-10 text-gray-500 mb-4" />
              <h3 className="text-base font-medium text-white">No Evaluation Triggered</h3>
              <p className="text-gray-400 text-xs mt-1 max-w-sm">
                Select a candidate resume and job description, then run the evaluation pipeline to invoke local FAISS indexing and LLM reasoning.
              </p>
            </div>
          ) : (
            <div className="space-y-6 animate-fadeIn">
              {/* Header and Mode Selection */}
              <div className="flex items-center justify-between border-b border-white/5 pb-4">
                <div>
                  <h2 className="text-lg font-medium text-white">Recruiter Assessment Profile</h2>
                  <p className="text-xs text-gray-400 mt-1">
                    Analysis ID: #{report.analysis_id} • Candidate Resume ID: #{report.resume_id} • Job ID: #{report.jd_id}
                  </p>
                </div>
                
                <div className="flex items-center gap-3">
                  <button
                    onClick={handleDownloadJSON}
                    className="flex items-center gap-1 text-xs border border-white/10 rounded-lg px-2.5 py-1 text-gray-300 hover:text-white hover:bg-white/5 transition"
                    title="Download Report JSON"
                  >
                    <Download className="h-3.5 w-3.5" /> JSON Report
                  </button>

                  <div className="flex rounded-lg border border-white/10 bg-black/20 p-0.5">
                    <button
                      onClick={() => setViewMode('report')}
                      className={`px-3 py-1 text-xs font-semibold rounded-md transition ${
                        viewMode === 'report'
                          ? 'bg-indigo-600 text-white'
                          : 'text-gray-400 hover:text-white'
                      }`}
                    >
                      Hiring Report
                    </button>
                    <button
                      onClick={() => setViewMode('raw_rag')}
                      className={`px-3 py-1 text-xs font-semibold rounded-md transition ${
                        viewMode === 'raw_rag'
                          ? 'bg-indigo-600 text-white'
                          : 'text-gray-400 hover:text-white'
                      }`}
                    >
                      Raw RAG Matches
                    </button>
                  </div>
                </div>
              </div>

              {viewMode === 'report' ? (
                /* Rich Recruiter Fit Report */
                <div className="space-y-6">
                  {/* Top Stats Overview */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    {/* Compatibility score Circle */}
                    <div className="rounded-xl border border-white/5 bg-card p-5 flex flex-col items-center justify-center text-center">
                      <span className="text-[10px] text-gray-500 font-semibold uppercase tracking-wider mb-2">Overall Score</span>
                      <div className={`flex items-center justify-center w-24 h-24 rounded-full border-2 text-3xl font-extrabold ${getScoreColor(report.overall_score)}`}>
                        {report.overall_score}%
                      </div>
                      <span className="text-xs text-gray-400 mt-3 font-semibold uppercase tracking-wide">
                        {report.overall_score >= 75 ? 'Recommended Fit' : report.overall_score >= 60 ? 'Conditional Fit' : 'Not Recommended'}
                      </span>
                    </div>

                    {/* Algorithmic Sub-scores */}
                    <div className="md:col-span-2 rounded-xl border border-white/5 bg-card p-5 space-y-3.5">
                      <h3 className="text-xs font-semibold text-white uppercase tracking-wider border-b border-white/5 pb-2">
                        Explainable Metric Breakdown
                      </h3>
                      
                      <div className="space-y-2 text-xs text-gray-300">
                        {/* Requirement Coverage */}
                        <div className="space-y-1">
                          <div className="flex justify-between font-medium">
                            <span>Requirement Coverage (Weight 35%):</span>
                            <span className="font-semibold">{report.coverage_score}%</span>
                          </div>
                          <div className="w-full bg-white/5 rounded-full h-1.5 overflow-hidden">
                            <div className={`h-full ${getScoreBarBg(report.coverage_score)}`} style={{ width: `${report.coverage_score}%` }}></div>
                          </div>
                        </div>

                        {/* Experience Alignment */}
                        <div className="space-y-1">
                          <div className="flex justify-between font-medium">
                            <span>Experience Alignment (Weight 25%):</span>
                            <span className="font-semibold">{report.experience_score}%</span>
                          </div>
                          <div className="w-full bg-white/5 rounded-full h-1.5 overflow-hidden">
                            <div className={`h-full ${getScoreBarBg(report.experience_score)}`} style={{ width: `${report.experience_score}%` }}></div>
                          </div>
                        </div>

                        {/* Project Relevance */}
                        <div className="space-y-1">
                          <div className="flex justify-between font-medium">
                            <span>Project Relevance (Weight 20%):</span>
                            <span className="font-semibold">{report.project_score}%</span>
                          </div>
                          <div className="w-full bg-white/5 rounded-full h-1.5 overflow-hidden">
                            <div className={`h-full ${getScoreBarBg(report.project_score)}`} style={{ width: `${report.project_score}%` }}></div>
                          </div>
                        </div>

                        {/* Evidence Confidence */}
                        <div className="space-y-1">
                          <div className="flex justify-between font-medium">
                            <span>Evidence Confidence (Weight 15%):</span>
                            <span className="font-semibold">{report.confidence_score}%</span>
                          </div>
                          <div className="w-full bg-white/5 rounded-full h-1.5 overflow-hidden">
                            <div className={`h-full ${getScoreBarBg(report.confidence_score)}`} style={{ width: `${report.confidence_score}%` }}></div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Recommendation and Executive Summary */}
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                    {/* Recommendation Card */}
                    <div className="md:col-span-1 rounded-xl border border-white/5 bg-card p-5 flex flex-col items-center justify-center text-center">
                      <Award className="h-7 w-7 text-indigo-400 mb-2" />
                      <span className="text-[10px] text-gray-500 font-semibold uppercase tracking-wider mb-1">Recruiter recommendation</span>
                      <span className="text-xs font-bold text-indigo-300 leading-tight">
                        {report.recruiter_recommendation}
                      </span>
                    </div>

                    {/* Executive Summary Card */}
                    <div className="md:col-span-3 rounded-xl border border-white/5 bg-card p-5 space-y-2">
                      <div className="flex items-center gap-1.5 text-xs font-semibold text-indigo-400 uppercase tracking-wider">
                        <FileText className="h-4 w-4" /> Recruiter summary
                      </div>
                      <p className="text-sm text-gray-300 leading-relaxed font-sans font-normal">
                        {report.summary}
                      </p>
                    </div>
                  </div>

                  {/* Dynamic Strengths & Weaknesses */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Strengths Card */}
                    <div className="rounded-xl border border-white/5 bg-card p-5 space-y-3">
                      <h3 className="text-xs font-semibold text-emerald-400 uppercase tracking-wider border-b border-white/5 pb-1">
                        Candidate Strengths
                      </h3>
                      <ul className="space-y-2 text-xs text-gray-300 list-disc list-inside">
                        {report.strengths.map((str, sIdx) => (
                          <li key={sIdx} className="leading-relaxed">{str}</li>
                        ))}
                      </ul>
                    </div>

                    {/* Weaknesses Card */}
                    <div className="rounded-xl border border-white/5 bg-card p-5 space-y-3">
                      <h3 className="text-xs font-semibold text-rose-400 uppercase tracking-wider border-b border-white/5 pb-1">
                        Candidate Weaknesses / Gaps
                      </h3>
                      <ul className="space-y-2 text-xs text-gray-300 list-disc list-inside">
                        {report.weaknesses.map((weak, wIdx) => (
                          <li key={wIdx} className="leading-relaxed">{weak}</li>
                        ))}
                      </ul>
                    </div>
                  </div>

                  {/* Skill Relationships (Module 5) */}
                  <div className="rounded-xl border border-white/5 bg-card p-5 space-y-3">
                    <div className="flex items-center gap-1.5 text-xs font-semibold text-indigo-400 uppercase tracking-wider">
                      <Brain className="h-4 w-4" /> Transferable Skill Relationships
                    </div>
                    <div className="space-y-2.5">
                      {report.skill_relationships.map((rel, rIdx) => (
                        <p key={rIdx} className="text-xs text-gray-300 bg-black/20 p-2.5 rounded-lg border border-white/5 leading-relaxed">
                          {rel}
                        </p>
                      ))}
                      {report.skill_relationships.length === 0 && (
                        <p className="text-xs text-gray-500 italic">No transferable skill dependencies detected.</p>
                      )}
                    </div>
                  </div>

                  {/* Requirement-by-Requirement Fit Cards */}
                  <div className="rounded-xl border border-white/5 bg-card p-6 space-y-4">
                    <h3 className="text-sm font-semibold text-white border-b border-white/5 pb-2">Skill Evidence Analysis</h3>
                    <div className="divide-y divide-white/5">
                      {report.requirements.map((req, idx) => (
                        <div key={idx} className="py-4 first:pt-0 last:pb-0 space-y-2.5">
                          <div className="flex justify-between items-start">
                            <div>
                              <span className="font-semibold text-sm text-gray-100 uppercase tracking-wide">
                                {req.requirement}
                              </span>
                              <span className="ml-2.5 inline-flex items-center rounded bg-white/5 border border-white/10 px-1.5 py-0.5 text-[10px] font-medium text-gray-400">
                                {req.category}
                              </span>
                            </div>
                            
                            <div className="flex items-center gap-2">
                              <span className="text-[10px] text-gray-400 font-medium">Confidence: {req.confidence}%</span>
                              {getStatusBadge(req.status)}
                            </div>
                          </div>
                          
                          <div className="space-y-1">
                            <p className="text-xs text-gray-300">
                              <span className="text-indigo-400 font-semibold">Relevance:</span> {req.explanation}
                            </p>
                            {req.limitations && req.limitations !== 'None' && (
                              <p className="text-xs text-gray-400">
                                <span className="text-rose-400 font-semibold">Limitations:</span> {req.limitations}
                              </p>
                            )}
                          </div>

                          {req.matched_evidence && (
                            <div className="rounded-lg bg-black/40 border border-white/5 p-3 text-xs font-mono text-indigo-300 leading-relaxed">
                              "{req.matched_evidence}"
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Missing Skills & Roadmaps */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Missing Skills Cards */}
                    <div className="rounded-xl border border-white/5 bg-card p-5 space-y-4">
                      <h3 className="text-sm font-semibold text-white border-b border-white/5 pb-2">
                        Missing Skills
                      </h3>
                      {report.missing_skills.length === 0 ? (
                        <p className="text-xs text-emerald-400 bg-emerald-500/5 border border-emerald-500/10 p-3 rounded-lg">
                          Excellent! Candidate covers all required technologies structurally.
                        </p>
                      ) : (
                        <div className="flex flex-wrap gap-2">
                          {report.missing_skills.map((skill, sIdx) => (
                            <span key={sIdx} className="rounded px-2.5 py-1 text-xs font-semibold bg-rose-500/15 border border-rose-500/30 text-rose-400 uppercase">
                              {skill}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>

                    {/* Learning Roadmap details */}
                    <div className="rounded-xl border border-white/5 bg-card p-5 space-y-4">
                      <div className="flex items-center gap-1.5 text-sm font-semibold text-indigo-400">
                        <BookOpen className="h-4 w-4" /> Personalized Learning Roadmap
                      </div>
                      <p className="text-xs text-gray-400">
                        Dynamic upskilling roadmaps estimated by analyzing technology overlap and missing concepts:
                      </p>
                      
                      <div className="space-y-3">
                        {report.learning_roadmap.map((item, rIdx) => (
                          <div key={rIdx} className="border-l-2 border-indigo-500/40 pl-3 space-y-0.5">
                            <div className="flex justify-between items-center text-xs">
                              <span className="font-semibold text-gray-200 uppercase">{item.skill}</span>
                              <span className="text-[10px] bg-indigo-500/10 text-indigo-300 border border-indigo-500/20 rounded px-1.5 py-0.5 font-semibold">
                                {item.estimated_time}
                              </span>
                            </div>
                            <p className="text-[11px] text-gray-400">{item.reason}</p>
                          </div>
                        ))}
                        {report.learning_roadmap.length === 0 && (
                          <p className="text-xs text-gray-500 italic">No upskilling items required.</p>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Interview Questions Validation Guide */}
                  <div className="rounded-xl border border-white/5 bg-card p-5 space-y-4">
                    <h3 className="text-sm font-semibold text-white border-b border-white/5 pb-2">
                      Interview Questions (Depth vs Verification)
                    </h3>
                    <p className="text-xs text-gray-400">
                      Recruiter guide to validate candidate's credentials specifically on low-confidence or weakly supported claims:
                    </p>
                    <div className="space-y-3">
                      {report.interview_questions.map((question, qIdx) => (
                        <div key={qIdx} className="flex gap-3 bg-black/20 p-3 rounded-lg border border-white/5 text-xs text-gray-300">
                          <span className="flex items-center justify-center shrink-0 w-5 h-5 rounded-full bg-indigo-600 text-white font-semibold text-[10px]">
                            {qIdx + 1}
                          </span>
                          <p className="leading-relaxed font-sans font-normal">{question}</p>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Rejection Email Template Preview */}
                  {report.rejection_email && (
                    <div className="rounded-xl border border-white/5 bg-card p-5 space-y-4">
                      <div className="flex justify-between items-center border-b border-white/5 pb-2">
                        <div className="flex items-center gap-2 text-sm font-semibold text-rose-400">
                          <Mail className="h-4 w-4" /> Recruiter Email Draft (Dynamic Rejection)
                        </div>
                        <button
                          onClick={handleCopyEmail}
                          className="flex items-center gap-1.5 text-xs border border-white/10 rounded-lg px-2.5 py-1 text-gray-400 hover:text-white hover:bg-white/5 transition"
                        >
                          {copiedEmail ? (
                            <>
                              <Check className="h-3 w-3 text-emerald-400" /> Copied!
                            </>
                          ) : (
                            <>
                              <Copy className="h-3 w-3" /> Copy Draft
                            </>
                          )}
                        </button>
                      </div>
                      
                      <pre className="rounded-lg bg-black/40 border border-white/5 p-4 text-xs text-gray-400 overflow-x-auto whitespace-pre-wrap font-mono leading-relaxed max-h-[300px]">
                        {report.rejection_email}
                      </pre>
                    </div>
                  )}
                </div>
              ) : (
                /* Raw RAG matches display */
                <div className="space-y-5">
                  <p className="text-xs text-gray-400">
                    Here are the semantic chunks matching the Job Description requirements extracted directly from the FAISS database index:
                  </p>
                  {report.retrieval_results.map((result, idx) => (
                    <div
                      key={idx}
                      className="rounded-xl border border-white/5 bg-card p-5 space-y-4 hover:border-white/10 transition"
                    >
                      <div className="flex items-center justify-between">
                        <span className="inline-flex items-center rounded-lg bg-indigo-500/10 px-2.5 py-1 text-xs font-semibold text-indigo-400 border border-indigo-500/20">
                          {result.requirement}
                        </span>
                        <span className="text-[10px] text-gray-500 font-semibold uppercase tracking-wider">
                          {result.matches.length} matches
                        </span>
                      </div>

                      {result.matches.length === 0 ? (
                        <div className="text-xs text-gray-500 bg-black/25 rounded-lg p-3 italic">
                          No direct semantic match found in resume sections.
                        </div>
                      ) : (
                        <div className="space-y-3">
                          {result.matches.map((match, mIdx) => (
                            <div
                              key={mIdx}
                              className="rounded-lg bg-black/30 p-3.5 border border-white/5 space-y-2 text-xs"
                            >
                              <div className="flex justify-between items-center text-[10px] font-semibold">
                                <div className="flex gap-2">
                                  <span className="text-indigo-300">
                                    Section: {match.section}
                                  </span>
                                  <span className="text-gray-500">•</span>
                                  <span className="text-cyan-400">
                                    Page {match.page}
                                  </span>
                                </div>
                                <span className="px-2 py-0.5 rounded border text-gray-400 border-white/10 bg-white/5">
                                  Score: {match.score}
                                </span>
                              </div>
                              <p className="text-gray-300 leading-relaxed font-sans font-normal whitespace-pre-wrap">
                                "{match.chunk}"
                              </p>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
