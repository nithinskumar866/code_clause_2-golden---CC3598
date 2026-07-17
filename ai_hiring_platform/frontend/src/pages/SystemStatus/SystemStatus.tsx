import React from 'react';
import { Cpu, Database, Eye, Users, ShieldAlert } from 'lucide-react';
import { PageHeader } from '../../components/ui/PageHeader';

export const SystemStatus: React.FC = () => {
  return (
    <div className="space-y-8 animate-fadeIn">
      <PageHeader
        title="System Status"
        description="Configuration and health of the core RAG and agent subsystems."
      />

      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        {/* LLM Provider */}
        <div className="rounded-xl border border-white/5 bg-card p-6 space-y-4">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2.5 bg-amber-500/10 rounded-lg text-amber-400">
                <Cpu className="h-6 w-6" />
              </div>
              <div>
                <h3 className="font-semibold text-white">LLM Provider</h3>
                <p className="text-xs text-gray-500 mt-0.5">Model reasoning engine</p>
              </div>
            </div>
            <span className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium border border-amber-500/30 bg-amber-500/10 text-amber-400">
              Not Configured
            </span>
          </div>
          <div className="border-t border-white/5 pt-4 text-xs text-gray-400 space-y-2">
            <div className="flex justify-between">
              <span>Selected Provider:</span>
              <span className="font-medium text-gray-300">None (Sprint 3)</span>
            </div>
            <div className="flex justify-between">
              <span>API Connection:</span>
              <span className="font-medium text-gray-300">Pending Setup</span>
            </div>
          </div>
        </div>

        {/* Vector Store */}
        <div className="rounded-xl border border-white/5 bg-card p-6 space-y-4">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2.5 bg-emerald-500/10 rounded-lg text-emerald-400">
                <Database className="h-6 w-6" />
              </div>
              <div>
                <h3 className="font-semibold text-white">Vector Store</h3>
                <p className="text-xs text-gray-500 mt-0.5">FAISS Vector database</p>
              </div>
            </div>
            <span className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium border border-emerald-500/30 bg-emerald-500/10 text-emerald-400">
              Active
            </span>
          </div>
          <div className="border-t border-white/5 pt-4 text-xs text-gray-400 space-y-2">
            <div className="flex justify-between">
              <span>Store Type:</span>
              <span className="font-medium text-gray-300">FAISS Index (Local)</span>
            </div>
            <div className="flex justify-between">
              <span>Indexed Vectors:</span>
              <span className="font-medium text-gray-300">Stored on Disk</span>
            </div>
          </div>
        </div>

        {/* Embedding Model */}
        <div className="rounded-xl border border-white/5 bg-card p-6 space-y-4">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2.5 bg-emerald-500/10 rounded-lg text-emerald-400">
                <Eye className="h-6 w-6" />
              </div>
              <div>
                <h3 className="font-semibold text-white">Embedding Model</h3>
                <p className="text-xs text-gray-500 mt-0.5">Semantic representation</p>
              </div>
            </div>
            <span className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium border border-emerald-500/30 bg-emerald-500/10 text-emerald-400">
              Active
            </span>
          </div>
          <div className="border-t border-white/5 pt-4 text-xs text-gray-400 space-y-2">
            <div className="flex justify-between">
              <span>Model Name:</span>
              <span className="font-medium text-gray-300">BAAI/bge-small-en-v1.5</span>
            </div>
            <div className="flex justify-between">
              <span>Execution:</span>
              <span className="font-medium text-gray-300">Local (LlamaIndex)</span>
            </div>
          </div>
        </div>

        {/* Agent Net Status */}
        <div className="rounded-xl border border-white/5 bg-card p-6 space-y-4">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2.5 bg-amber-500/10 rounded-lg text-amber-400">
                <Users className="h-6 w-6" />
              </div>
              <div>
                <h3 className="font-semibold text-white">Agent Status</h3>
                <p className="text-xs text-gray-500 mt-0.5">Collaboration state</p>
              </div>
            </div>
            <span className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium border border-amber-500/30 bg-amber-500/10 text-amber-400">
              Partially Active
            </span>
          </div>
          <div className="border-t border-white/5 pt-4 text-xs text-gray-400 space-y-2">
            <div className="flex justify-between">
              <span>Candidate Intelligence Agent:</span>
              <span className="font-medium text-emerald-400">Active (Knowledge Retrieval Engine)</span>
            </div>
            <div className="flex justify-between">
              <span>Hiring Decision Agent:</span>
              <span className="font-medium text-gray-500">Uninitialized (Sprint 3)</span>
            </div>
          </div>
        </div>
      </div>

      <div className="rounded-xl border border-white/5 bg-card p-6 space-y-4">
        <div className="flex items-center gap-2">
          <ShieldAlert className="h-5 w-5 text-amber-500" />
          <h2 className="text-lg font-medium text-white">Platform Health</h2>
        </div>
        <p className="text-sm text-gray-400">
          The Candidate Intelligence Agent is running local RAG retrieval pipelines over candidate resumes. LLM reasoning features (hiring fit scores, interview questions, email generation) will be activated in Sprint 3 once a reasoning provider (OpenAI, Anthropic, or Ollama) is configured.
        </p>
      </div>
    </div>
  );
};
