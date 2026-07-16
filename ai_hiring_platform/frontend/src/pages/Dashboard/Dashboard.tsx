import React, { useEffect, useState } from 'react';
import { Server, Database, BrainCircuit, ShieldAlert, Activity } from 'lucide-react';
import axios from 'axios';

interface HealthStatus {
  backend: 'Connected' | 'Disconnected' | 'Checking';
  database: 'Connected' | 'Disconnected' | 'Checking';
}

export const Dashboard: React.FC = () => {
  const [status, setStatus] = useState<HealthStatus>({
    backend: 'Checking',
    database: 'Checking',
  });

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await axios.get('http://localhost:8000/api/v1/health');
        if (response.data && response.data.success) {
          setStatus({
            backend: 'Connected',
            database: response.data.data.database === 'connected' ? 'Connected' : 'Disconnected',
          });
        } else {
          setStatus({ backend: 'Disconnected', database: 'Disconnected' });
        }
      } catch (error) {
        console.error('Failed to reach backend health endpoint:', error);
        setStatus({ backend: 'Disconnected', database: 'Disconnected' });
      }
    };

    checkHealth();
    // Poll every 10 seconds
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (val: string) => {
    if (val === 'Connected') return 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30';
    if (val === 'Disconnected') return 'text-rose-400 bg-rose-500/10 border-rose-500/30';
    return 'text-amber-400 bg-amber-500/10 border-amber-500/30';
  };

  const getStatusDot = (val: string) => {
    if (val === 'Connected') return 'bg-emerald-500 shadow-emerald-500/50';
    if (val === 'Disconnected') return 'bg-rose-500 shadow-rose-500/50';
    return 'bg-amber-500 animate-pulse';
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-white">Dashboard Overview</h1>
        <p className="mt-2 text-sm text-gray-400">
          Real-time status tracking of the recruitment agent network and platform health.
        </p>
      </div>

      {/* Status Cards */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        {/* Backend status */}
        <div className="relative overflow-hidden rounded-xl border border-white/5 bg-card p-5 transition hover:border-white/10">
          <div className="flex items-center justify-between">
            <div className="p-2.5 rounded-lg bg-indigo-500/10 text-indigo-400">
              <Server className="h-6 w-6" />
            </div>
            <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium border ${getStatusColor(status.backend)}`}>
              <span className={`mr-1.5 h-1.5 w-1.5 rounded-full ${getStatusDot(status.backend)}`} />
              {status.backend}
            </span>
          </div>
          <div className="mt-4">
            <h3 className="text-sm font-medium text-gray-400">Backend API</h3>
            <p className="text-xl font-semibold text-white mt-1">FastAPI Service</p>
          </div>
        </div>

        {/* Database status */}
        <div className="relative overflow-hidden rounded-xl border border-white/5 bg-card p-5 transition hover:border-white/10">
          <div className="flex items-center justify-between">
            <div className="p-2.5 rounded-lg bg-cyan-500/10 text-cyan-400">
              <Database className="h-6 w-6" />
            </div>
            <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium border ${getStatusColor(status.database)}`}>
              <span className={`mr-1.5 h-1.5 w-1.5 rounded-full ${getStatusDot(status.database)}`} />
              {status.database}
            </span>
          </div>
          <div className="mt-4">
            <h3 className="text-sm font-medium text-gray-400">Database</h3>
            <p className="text-xl font-semibold text-white mt-1">SQLite (SQLAlchemy)</p>
          </div>
        </div>

        {/* Resume Agent status */}
        <div className="relative overflow-hidden rounded-xl border border-white/5 bg-card p-5 transition hover:border-white/10">
          <div className="flex items-center justify-between">
            <div className="p-2.5 rounded-lg bg-pink-500/10 text-pink-400">
              <BrainCircuit className="h-6 w-6" />
            </div>
            <span className="inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium border border-amber-500/30 bg-amber-500/10 text-amber-400">
              <span className="mr-1.5 h-1.5 w-1.5 rounded-full bg-amber-500" />
              Not Configured
            </span>
          </div>
          <div className="mt-4">
            <h3 className="text-sm font-medium text-gray-400">Resume Intel Agent</h3>
            <p className="text-xl font-semibold text-white mt-1">LlamaIndex RAG</p>
          </div>
        </div>

        {/* Evaluator Agent status */}
        <div className="relative overflow-hidden rounded-xl border border-white/5 bg-card p-5 transition hover:border-white/10">
          <div className="flex items-center justify-between">
            <div className="p-2.5 rounded-lg bg-emerald-500/10 text-emerald-400">
              <ShieldAlert className="h-6 w-6" />
            </div>
            <span className="inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium border border-amber-500/30 bg-amber-500/10 text-amber-400">
              <span className="mr-1.5 h-1.5 w-1.5 rounded-full bg-amber-500" />
              Not Configured
            </span>
          </div>
          <div className="mt-4">
            <h3 className="text-sm font-medium text-gray-400">Evaluator Agent</h3>
            <p className="text-xl font-semibold text-white mt-1">LLM Reasoning</p>
          </div>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="rounded-xl border border-white/5 bg-card p-6">
        <div className="flex items-center justify-between border-b border-white/5 pb-4">
          <div className="flex items-center space-x-2">
            <Activity className="h-5 w-5 text-indigo-400" />
            <h2 className="text-lg font-medium text-white">Recent Activity</h2>
          </div>
        </div>
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <p className="text-gray-400 text-sm">No analysis history found.</p>
          <p className="text-gray-500 text-xs mt-1">Uploaded documents will appear here once analysis is configured in Sprint 2.</p>
        </div>
      </div>
    </div>
  );
};
