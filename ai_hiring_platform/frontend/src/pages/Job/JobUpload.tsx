import React, { useState, useRef } from 'react';
import { UploadCloud, FileText, CheckCircle2, AlertCircle, RefreshCw } from 'lucide-react';
import axios from 'axios';

interface UploadResult {
  id: number;
  filename: string;
  upload_time: string;
  status: string;
}

export const JobUpload: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [isDragActive, setIsDragActive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<UploadResult | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragActive(true);
    } else if (e.type === "dragleave") {
      setIsDragActive(false);
    }
  };

  const validateFile = (file: File): boolean => {
    const ext = file.name.split('.').pop()?.toLowerCase();
    if (ext !== 'pdf' && ext !== 'docx') {
      setError("Unsupported file format. Only PDF and DOCX files are allowed.");
      setFile(null);
      return false;
    }
    setError(null);
    return true;
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (validateFile(droppedFile)) {
        setFile(droppedFile);
        setResult(null);
      }
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      if (validateFile(selectedFile)) {
        setFile(selectedFile);
        setResult(null);
      }
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8000/api/v1/job/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      if (response.data && response.data.success) {
        setResult(response.data.data);
      } else {
        setError(response.data?.message || "Upload failed");
      }
    } catch (err: any) {
      console.error(err);
      const detailMsg = err.response?.data?.error?.details || err.message || "Failed to reach server";
      setError(detailMsg);
    } finally {
      setLoading(false);
    }
  };

  const resetUpload = () => {
    setFile(null);
    setResult(null);
    setError(null);
  };

  const formatBytes = (bytes: number, decimals = 2) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-white">Upload Job Descriptions</h1>
        <p className="mt-2 text-sm text-gray-400">
          Upload Job Descriptions in PDF or DOCX format to store metadata. Parsing and indexing are executed in Sprint 2.
        </p>
      </div>

      <div className="max-w-2xl">
        {!result ? (
          <div className="space-y-6">
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
                onChange={handleChange}
              />
              <UploadCloud className={`h-12 w-12 mb-4 transition ${isDragActive ? 'text-indigo-400' : 'text-gray-400'}`} />
              <p className="text-sm text-gray-200 font-medium">
                Drag and drop your job description file here
              </p>
              <p className="text-xs text-gray-400 mt-1">
                or click to browse from local directories
              </p>
              <p className="text-xs text-gray-500 mt-4">
                Supported formats: PDF, DOCX (Max size 10MB)
              </p>
            </div>

            {/* Error alerts */}
            {error && (
              <div className="flex items-center gap-3 rounded-lg border border-rose-500/20 bg-rose-500/10 p-4 text-sm text-rose-400">
                <AlertCircle className="h-5 w-5 shrink-0" />
                <span>{error}</span>
              </div>
            )}

            {/* Selected File Details */}
            {file && (
              <div className="flex items-center justify-between rounded-xl border border-white/5 bg-card p-4">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-indigo-500/10 rounded-lg text-indigo-400">
                    <FileText className="h-6 w-6" />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-white">{file.name}</p>
                    <p className="text-xs text-gray-400">{formatBytes(file.size)}</p>
                  </div>
                </div>
                
                <button
                  onClick={handleUpload}
                  disabled={loading}
                  className="flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition"
                >
                  {loading ? (
                    <>
                      <RefreshCw className="h-4 w-4 animate-spin" />
                      Uploading...
                    </>
                  ) : (
                    'Upload File'
                  )}
                </button>
              </div>
            )}
          </div>
        ) : (
          /* Success Result Card */
          <div className="rounded-xl border border-emerald-500/20 bg-card p-6 space-y-6">
            <div className="flex items-center gap-3 text-emerald-400 font-medium">
              <CheckCircle2 className="h-6 w-6" />
              <span>Job Description Saved Successfully</span>
            </div>

            <div className="grid grid-cols-2 gap-4 rounded-lg bg-black/30 p-4 border border-white/5">
              <div>
                <p className="text-xs text-gray-400 font-medium">Database ID</p>
                <p className="text-sm font-semibold text-white mt-0.5">#{result.id}</p>
              </div>
              <div>
                <p className="text-xs text-gray-400 font-medium">Document Status</p>
                <p className="text-sm font-semibold text-emerald-400 mt-0.5">{result.status}</p>
              </div>
              <div className="col-span-2">
                <p className="text-xs text-gray-400 font-medium">File Name</p>
                <p className="text-sm font-semibold text-white mt-0.5 truncate">{result.filename}</p>
              </div>
              <div className="col-span-2">
                <p className="text-xs text-gray-400 font-medium">Upload Time</p>
                <p className="text-sm font-semibold text-white mt-0.5">
                  {new Date(result.upload_time).toLocaleString()}
                </p>
              </div>
            </div>

            <div className="flex gap-3">
              <button
                onClick={resetUpload}
                className="rounded-lg border border-white/10 px-4 py-2 text-sm font-semibold text-white hover:bg-white/5 transition"
              >
                Upload Another Job Description
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
