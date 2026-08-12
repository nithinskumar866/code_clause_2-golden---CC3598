import React, { useState, useRef } from 'react';
import { UploadCloud, FileText, CheckCircle2, RefreshCw, XCircle, AlertCircle, Trash2 } from 'lucide-react';
import { api } from '../../api/client';
import type { BulkUploadResult, BulkUploadFileResult } from '../../types';
import { PageHeader } from '../../components/ui/PageHeader';
import { Banner } from '../../components/ui/Banner';
import { useToast } from '../../components/ui/toast-context';

type FileStatus = 'pending' | 'uploading' | 'success' | 'error';

interface FileEntry {
  file: File;
  status: FileStatus;
  error?: string;
  id?: number;
}

export const JobUpload: React.FC = () => {
  const [files, setFiles] = useState<FileEntry[]>([]);
  const [isDragActive, setIsDragActive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [bulkResult, setBulkResult] = useState<BulkUploadResult | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const toast = useToast();

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragActive(true);
    } else if (e.type === "dragleave") {
      setIsDragActive(false);
    }
  };

  const validateAndAddFiles = (newFiles: FileList | File[]) => {
    const validFiles: FileEntry[] = [];
    const invalidNames: string[] = [];

    Array.from(newFiles).forEach((file) => {
      const ext = file.name.split('.').pop()?.toLowerCase();
      if (ext === 'pdf' || ext === 'docx') {
        if (!files.some((f) => f.file.name === file.name && f.file.size === file.size)) {
          validFiles.push({ file, status: 'pending' });
        }
      } else {
        invalidNames.push(file.name);
      }
    });

    if (invalidNames.length > 0) {
      setError(`Unsupported files skipped: ${invalidNames.join(', ')}. Only PDF and DOCX allowed.`);
    } else {
      setError(null);
    }

    if (validFiles.length > 0) {
      setFiles((prev) => [...prev, ...validFiles]);
      setBulkResult(null);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndAddFiles(e.dataTransfer.files);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files.length > 0) {
      validateAndAddFiles(e.target.files);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const clearAll = () => {
    setFiles([]);
    setBulkResult(null);
    setError(null);
  };

  const handleUpload = async () => {
    const pendingFiles = files.filter((f) => f.status === 'pending');
    if (pendingFiles.length === 0) return;

    setLoading(true);
    setError(null);

    setFiles((prev) =>
      prev.map((f) => (f.status === 'pending' ? { ...f, status: 'uploading' as FileStatus } : f))
    );

    if (pendingFiles.length === 1) {
      const formData = new FormData();
      formData.append('file', pendingFiles[0].file);

      try {
        const response = await api.post('/job/upload', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
        if (response.data?.success) {
          setFiles((prev) =>
            prev.map((f) =>
              f.file === pendingFiles[0].file
                ? { ...f, status: 'success' as FileStatus, id: response.data.data.id }
                : f
            )
          );
          toast.success('Job description uploaded successfully', response.data.data.filename);
        } else {
          const msg = response.data?.message || 'Upload failed';
          setFiles((prev) =>
            prev.map((f) =>
              f.file === pendingFiles[0].file ? { ...f, status: 'error' as FileStatus, error: msg } : f
            )
          );
          toast.error(msg);
        }
      } catch (err: any) {
        const msg = err.response?.data?.error?.details || err.message || 'Upload failed';
        setFiles((prev) =>
          prev.map((f) =>
            f.file === pendingFiles[0].file ? { ...f, status: 'error' as FileStatus, error: msg } : f
          )
        );
        toast.error(msg);
      }
    } else {
      const formData = new FormData();
      pendingFiles.forEach((f) => formData.append('files', f.file));

      try {
        const response = await api.post('/job/upload-bulk', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });

        if (response.data?.success) {
          const result: BulkUploadResult = response.data.data;
          setBulkResult(result);

          const resultMap = new Map<string, BulkUploadFileResult>();
          result.results.forEach((r) => resultMap.set(r.filename, r));

          setFiles((prev) =>
            prev.map((f) => {
              const r = resultMap.get(f.file.name);
              if (r) {
                return {
                  ...f,
                  status: (r.success ? 'success' : 'error') as FileStatus,
                  error: r.error,
                  id: r.data?.id,
                };
              }
              return f;
            })
          );

          toast.success(
            'Bulk upload complete',
            `${result.success_count} uploaded, ${result.failed_count} failed`
          );
        }
      } catch (err: any) {
        const msg = err.response?.data?.error?.details || err.message || 'Bulk upload failed';
        setError(msg);
        setFiles((prev) =>
          prev.map((f) => (f.status === 'uploading' ? { ...f, status: 'error' as FileStatus, error: msg } : f))
        );
        toast.error(msg);
      }
    }

    setLoading(false);
  };

  const formatBytes = (bytes: number, decimals = 2) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  };

  const pendingCount = files.filter((f) => f.status === 'pending').length;
  const successCount = files.filter((f) => f.status === 'success').length;
  const errorCount = files.filter((f) => f.status === 'error').length;

  const statusIcon = (status: FileStatus) => {
    switch (status) {
      case 'success':
        return <CheckCircle2 className="h-4 w-4 text-emerald-400" />;
      case 'error':
        return <XCircle className="h-4 w-4 text-rose-400" />;
      case 'uploading':
        return <RefreshCw className="h-4 w-4 text-indigo-400 animate-spin" />;
      default:
        return <FileText className="h-4 w-4 text-gray-400" />;
    }
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      <PageHeader
        title="Job Descriptions"
        description="Upload job descriptions (PDF or DOCX). Select multiple files or drag to bulk-upload."
      />

      <div className="max-w-3xl space-y-6">
        {/* Drag & Drop Box */}
        <div
          onDragEnter={handleDrag}
          onDragOver={handleDrag}
          onDragLeave={handleDrag}
          onDrop={handleDrop}
          onClick={triggerFileInput}
          className={`flex flex-col items-center justify-center border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition duration-200 ${
            isDragActive
              ? 'border-indigo-500 bg-indigo-500/5'
              : 'border-white/10 bg-card hover:border-white/20'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            accept=".pdf,.docx"
            multiple
            onChange={handleChange}
          />
          <UploadCloud className={`h-12 w-12 mb-4 transition ${isDragActive ? 'text-indigo-400' : 'text-gray-400'}`} />
          <p className="text-sm text-gray-200 font-medium">
            Drag and drop job description files here
          </p>
          <p className="text-xs text-gray-400 mt-1">
            or click to browse — select multiple files for bulk upload
          </p>
          <p className="text-xs text-gray-500 mt-4">
            Supported formats: PDF, DOCX (Max size 10MB each)
          </p>
        </div>

        {/* Error alerts */}
        {error && <Banner variant="error">{error}</Banner>}

        {/* Bulk upload outcome — failures are named, since that is the only part a
            recruiter needs to act on after a large batch. */}
        {bulkResult && (
          <div className="rounded-xl border border-white/10 bg-card p-4">
            <div className="flex flex-wrap items-center gap-3">
              <span className="inline-flex items-center gap-1.5 text-sm font-semibold text-emerald-400">
                <CheckCircle2 className="h-4 w-4" />
                {bulkResult.success_count} uploaded
              </span>
              {bulkResult.failed_count > 0 && (
                <span className="inline-flex items-center gap-1.5 text-sm font-semibold text-rose-400">
                  <AlertCircle className="h-4 w-4" />
                  {bulkResult.failed_count} failed
                </span>
              )}
              <span className="text-xs text-gray-500">of {bulkResult.total} files</span>
            </div>

            {bulkResult.failed_count > 0 && (
              <ul className="mt-3 space-y-1 border-t border-white/5 pt-3">
                {bulkResult.results
                  .filter((r) => !r.success)
                  .map((r) => (
                    <li key={r.filename} className="flex items-start gap-2 text-xs text-gray-400">
                      <XCircle className="mt-0.5 h-3 w-3 shrink-0 text-rose-400" />
                      <span className="min-w-0">
                        <span className="font-medium text-gray-300">{r.filename}</span>
                        {r.error && <span className="text-gray-500"> — {r.error}</span>}
                      </span>
                    </li>
                  ))}
              </ul>
            )}
          </div>
        )}

        {/* File list */}
        {files.length > 0 && (
          <div className="space-y-3">
            {/* File list header */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <h3 className="text-sm font-semibold text-white">{files.length} file{files.length !== 1 ? 's' : ''} selected</h3>
                {successCount > 0 && (
                  <span className="text-[10px] font-semibold text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 rounded-full px-2 py-0.5">
                    {successCount} uploaded
                  </span>
                )}
                {errorCount > 0 && (
                  <span className="text-[10px] font-semibold text-rose-400 bg-rose-500/10 border border-rose-500/20 rounded-full px-2 py-0.5">
                    {errorCount} failed
                  </span>
                )}
              </div>
              <button
                onClick={clearAll}
                className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-300 transition"
              >
                <Trash2 className="h-3 w-3" /> Clear all
              </button>
            </div>

            {/* File entries */}
            <div className="rounded-xl border border-white/5 bg-card divide-y divide-white/5 max-h-80 overflow-y-auto">
              {files.map((entry, i) => (
                <div key={`${entry.file.name}-${i}`} className="flex items-center gap-3 px-4 py-2.5">
                  {statusIcon(entry.status)}
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium text-white truncate">{entry.file.name}</p>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] text-gray-500">{formatBytes(entry.file.size)}</span>
                      {entry.error && (
                        <span className="text-[10px] text-rose-400 truncate">{entry.error}</span>
                      )}
                      {entry.id && (
                        <span className="text-[10px] text-emerald-400">ID #{entry.id}</span>
                      )}
                    </div>
                  </div>
                  {entry.status === 'pending' && (
                    <button
                      onClick={() => removeFile(i)}
                      className="text-gray-500 hover:text-gray-300 transition p-1"
                    >
                      <XCircle className="h-3.5 w-3.5" />
                    </button>
                  )}
                </div>
              ))}
            </div>

            {/* Upload button */}
            {pendingCount > 0 && (
              <button
                onClick={handleUpload}
                disabled={loading}
                className="flex items-center gap-2 rounded-lg bg-indigo-600 px-5 py-2.5 text-sm font-semibold text-white hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition w-full justify-center"
              >
                {loading ? (
                  <>
                    <RefreshCw className="h-4 w-4 animate-spin" />
                    Uploading {pendingCount} file{pendingCount !== 1 ? 's' : ''}...
                  </>
                ) : (
                  <>
                    <UploadCloud className="h-4 w-4" />
                    Upload {pendingCount} file{pendingCount !== 1 ? 's' : ''}
                  </>
                )}
              </button>
            )}

            {/* Completion summary */}
            {pendingCount === 0 && files.length > 0 && (
              <div className="rounded-xl border border-emerald-500/20 bg-card p-4 flex items-center gap-3">
                <CheckCircle2 className="h-5 w-5 text-emerald-400 shrink-0" />
                <div>
                  <p className="text-sm font-medium text-emerald-400">Upload Complete</p>
                  <p className="text-xs text-gray-400 mt-0.5">
                    {successCount} uploaded successfully{errorCount > 0 ? `, ${errorCount} failed` : ''}
                  </p>
                </div>
                <button
                  onClick={clearAll}
                  className="ml-auto rounded-lg border border-white/10 px-3 py-1.5 text-xs font-semibold text-white hover:bg-white/5 transition"
                >
                  Upload More
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
