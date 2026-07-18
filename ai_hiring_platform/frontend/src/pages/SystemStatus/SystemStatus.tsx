import { useEffect, useState, type FC, type ReactNode } from 'react';
import { Cpu, Database, Eye, Users, ShieldCheck, Server } from 'lucide-react';
import { api } from '../../api/client';
import { PageHeader } from '../../components/ui/PageHeader';

type Health = 'Connected' | 'Disconnected' | 'Checking';

const toneClass = (h: Health): string =>
  h === 'Connected'
    ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-400'
    : h === 'Disconnected'
      ? 'border-rose-500/30 bg-rose-500/10 text-rose-400'
      : 'border-amber-500/30 bg-amber-500/10 text-amber-400';

interface CardProps {
  icon: ReactNode;
  iconClass?: string;
  title: string;
  subtitle: string;
  status: string;
  statusClass: string;
  rows: Array<{ label: string; value: string; valueClass?: string }>;
}

const StatusCard: FC<CardProps> = ({ icon, iconClass, title, subtitle, status, statusClass, rows }) => (
  <div className="rounded-xl border border-white/5 bg-card p-6 space-y-4">
    <div className="flex items-start justify-between">
      <div className="flex items-center gap-3">
        <div className={`p-2.5 rounded-lg ${iconClass ?? 'bg-indigo-500/10 text-indigo-400'}`}>{icon}</div>
        <div>
          <h3 className="font-semibold text-white">{title}</h3>
          <p className="text-xs text-gray-500 mt-0.5">{subtitle}</p>
        </div>
      </div>
      <span className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium ${statusClass}`}>
        {status}
      </span>
    </div>
    <div className="border-t border-white/5 pt-4 text-xs text-gray-400 space-y-2">
      {rows.map((r) => (
        <div key={r.label} className="flex justify-between">
          <span>{r.label}:</span>
          <span className={`font-medium ${r.valueClass ?? 'text-gray-300'}`}>{r.value}</span>
        </div>
      ))}
    </div>
  </div>
);

/**
 * Configuration + health of the RAG and agent subsystems. Backend + database
 * badges are polled live from `/health`; the local RAG stack (BGE embeddings,
 * FAISS) always runs on-device; reasoning is provider-agnostic with a
 * deterministic offline fallback, so the pipeline works with or without a key.
 */
export const SystemStatus: FC = () => {
  const [backend, setBackend] = useState<Health>('Checking');
  const [database, setDatabase] = useState<Health>('Checking');

  useEffect(() => {
    let active = true;
    const check = async () => {
      try {
        const res = await api.get('/health');
        if (!active) return;
        if (res.data?.success) {
          setBackend('Connected');
          setDatabase(res.data.data?.database === 'connected' ? 'Connected' : 'Disconnected');
        } else {
          setBackend('Disconnected');
          setDatabase('Disconnected');
        }
      } catch {
        if (!active) return;
        setBackend('Disconnected');
        setDatabase('Disconnected');
      }
    };
    check();
    const id = setInterval(check, 10000);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, []);

  return (
    <div className="space-y-8 animate-fadeIn">
      <PageHeader
        title="System Status"
        description="Configuration and health of the core RAG and agent subsystems."
      />

      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        <StatusCard
          icon={<Server className="h-6 w-6" />}
          title="Backend API"
          subtitle="FastAPI service"
          status={backend}
          statusClass={toneClass(backend)}
          rows={[
            { label: 'Framework', value: 'FastAPI (/api/v1)' },
            { label: 'Database', value: 'SQLite', valueClass: database === 'Connected' ? 'text-emerald-400' : 'text-gray-300' },
          ]}
        />

        <StatusCard
          icon={<Cpu className="h-6 w-6" />}
          iconClass="bg-amber-500/10 text-amber-400"
          title="LLM Provider"
          subtitle="Model reasoning engine"
          status="Provider-agnostic"
          statusClass="border-indigo-500/30 bg-indigo-500/10 text-indigo-400"
          rows={[
            { label: 'Supported', value: 'OpenAI · Anthropic · Gemini' },
            { label: 'Configuration', value: 'LLM_PROVIDER / LLM_MODEL (env)' },
            { label: 'Offline fallback', value: 'Deterministic engine', valueClass: 'text-emerald-400' },
          ]}
        />

        <StatusCard
          icon={<Eye className="h-6 w-6" />}
          iconClass="bg-emerald-500/10 text-emerald-400"
          title="Embedding Model"
          subtitle="Semantic representation"
          status="Active"
          statusClass="border-emerald-500/30 bg-emerald-500/10 text-emerald-400"
          rows={[
            { label: 'Model', value: 'BAAI/bge-small-en-v1.5' },
            { label: 'Execution', value: 'Local (LlamaIndex)' },
          ]}
        />

        <StatusCard
          icon={<Database className="h-6 w-6" />}
          iconClass="bg-emerald-500/10 text-emerald-400"
          title="Vector Store"
          subtitle="FAISS vector database"
          status="Active"
          statusClass="border-emerald-500/30 bg-emerald-500/10 text-emerald-400"
          rows={[
            { label: 'Store type', value: 'FAISS IndexFlatIP (local)' },
            { label: 'Index scope', value: 'Per-resume, persisted to disk' },
          ]}
        />

        <StatusCard
          icon={<Users className="h-6 w-6" />}
          iconClass="bg-indigo-500/10 text-indigo-400"
          title="Agent Network"
          subtitle="Two-agent LangGraph pipeline"
          status="Active"
          statusClass="border-emerald-500/30 bg-emerald-500/10 text-emerald-400"
          rows={[
            { label: 'Candidate Intelligence', value: 'Evidence retrieval', valueClass: 'text-emerald-400' },
            { label: 'Hiring Decision', value: 'Evidence reasoning', valueClass: 'text-emerald-400' },
          ]}
        />

        <StatusCard
          icon={<ShieldCheck className="h-6 w-6" />}
          iconClass="bg-emerald-500/10 text-emerald-400"
          title="Explainability"
          subtitle="Deterministic evidence signals"
          status="Active"
          statusClass="border-emerald-500/30 bg-emerald-500/10 text-emerald-400"
          rows={[
            { label: 'Authenticity check', value: 'Keyword-stuffing / over-claim' },
            { label: 'Scoring', value: 'Reproducible, algorithm-owned' },
          ]}
        />
      </div>

      <div className="rounded-xl border border-white/5 bg-card p-6 space-y-2">
        <div className="flex items-center gap-2">
          <ShieldCheck className="h-5 w-5 text-indigo-400" />
          <h2 className="text-lg font-medium text-white">Platform Health</h2>
        </div>
        <p className="text-sm text-gray-400">
          Both runtime agents are active: the Candidate Intelligence Agent runs the local RAG pipeline
          (BGE embeddings → FAISS retrieval → evidence assembly) and the Hiring Decision Agent reasons over
          that evidence to produce the explainable report. Reasoning is provider-agnostic — set a provider
          key to enable an LLM, or run fully offline on the deterministic engine. Every score is computed by
          reproducible algorithms, never by the LLM.
        </p>
      </div>
    </div>
  );
};
